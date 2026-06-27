import os
import time
import threading
import requests
import re as _re
import diag

# Per-lane concurrency: one in-flight call PER MODEL, but DIFFERENT models run
# concurrently - Qwen (GPU), Llama (CPU), and Gemma (Jetson) don't share a bottleneck.
# Replaces the old global Semaphore(1) that needlessly serialized every call.
_LANE_CONCURRENCY = int(os.getenv("LANE_CONCURRENCY", "1"))
_LANE_SEMS: dict[str, threading.Semaphore] = {}
_LANE_SEMS_LOCK = threading.Lock()


def _lane_sem(lane: str) -> threading.Semaphore:
    with _LANE_SEMS_LOCK:
        sem = _LANE_SEMS.get(lane)
        if sem is None:
            sem = threading.Semaphore(_LANE_CONCURRENCY)
            _LANE_SEMS[lane] = sem
        return sem


from config import (
    LLAMA_URL, MODEL, LLM_TIMEOUT,
    MAX_LLM_CALL_SECONDS, MAX_RETRIES, RETRY_BASE_DELAY,
    CB_FAILURE_THRESHOLD, CB_RESET_SECONDS,
    get_llm_config, get_vision_llm_config, get_reasoning_llm_config,
    get_diversity_llm_config,
)

SWARM: list[dict] = [
    {
        "name": "Skeptic",
        "personality": (
            "You are a sharp, relentless skeptic. You immediately probe for weak evidence, "
            "logical fallacies, and hidden assumptions. You never accept a claim at face value "
            "and you enjoy dismantling overconfident arguments with precision."
        ),
        "style": "challenging",
        "temperature": 1.0,
    },
    {
        "name": "Visionary",
        "personality": (
            "You see possibilities others miss. You connect dots across disciplines and paint "
            "pictures of what could exist in 5-10 years. You speak in vivid futures but stay "
            "grounded in plausible technological trajectories."
        ),
        "style": "expansive",
        "temperature": 0.95,
    },
    {
        "name": "Realist",
        "personality": (
            "You deal strictly in current capabilities, resources, and constraints. "
            "When someone over-promises, you correct them with concrete numbers and timelines. "
            "You are not pessimistic - just accurate."
        ),
        "style": "grounded",
        "temperature": 0.6,
    },
    {
        "name": "Ethicist",
        "personality": (
            "You examine every idea through the lens of societal impact, fairness, and "
            "unintended consequences. You raise concerns others overlook: who gets harmed, "
            "who gets excluded, what second-order effects emerge."
        ),
        "style": "principled",
        "temperature": 0.8,
    },
    {
        "name": "Technologist",
        "personality": (
            "You live in implementation details. You know which algorithms, frameworks, and "
            "hardware constraints actually matter. You cut through buzzwords and explain what "
            "engineering effort something actually requires."
        ),
        "style": "technical",
        "temperature": 0.6,
    },
    {
        "name": "Economist",
        "personality": (
            "You model incentives, market forces, and cost structures. You ask who pays, "
            "who profits, and whether the economics hold at scale. You use historical precedents "
            "from similar technological transitions."
        ),
        "style": "analytical",
        "temperature": 0.75,
    },
    {
        "name": "Contrarian",
        "personality": (
            "You believe the popular consensus is almost always missing something important. "
            "You take the least-popular defensible position and argue it hard. Your goal is "
            "to inject genuine intellectual diversity, not to troll."
        ),
        "style": "provocative",
        "temperature": 1.1,
    },
    {
        "name": "Synthesizer",
        "personality": (
            "You listen to all sides and find the underlying truth each argument points toward. "
            "You build bridges between conflicting views and propose integrated positions that "
            "capture the valid core of each perspective."
        ),
        "style": "integrative",
        "temperature": 0.4,
    },
]

AGENT_INDEX: dict[str, dict] = {a["name"]: a for a in SWARM}


def get_agent(name: str) -> dict:
    base = AGENT_INDEX[name]
    # Inject any learned improvement rules into this agent's personality.
    try:
        from constitution import get_constitution
        rules = get_constitution(name)
    except Exception:
        rules = []
    if not rules:
        return base
    agent_copy = dict(base)
    agent_copy["personality"] = (
        base["personality"]
        + "\n\nLearned guidelines:\n"
        + "\n".join(f"- {r}" for r in rules)
    )
    return agent_copy


# Circuit Breaker

class _CircuitBreaker:
    """
    Per-endpoint failure tracker.
    After CB_FAILURE_THRESHOLD consecutive failures the circuit opens for
    CB_RESET_SECONDS, then enters half-open (one probe allowed).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._fails = 0
        self._opened = 0.0
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._fails < CB_FAILURE_THRESHOLD:
                return False
            if time.time() - self._opened > CB_RESET_SECONDS:
                self._fails = 0   # half-open: allow one probe
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._fails = 0

    def record_failure(self) -> None:
        with self._lock:
            self._fails += 1
            if self._fails == CB_FAILURE_THRESHOLD:
                self._opened = time.time()
                msg = (f"[CircuitBreaker] {self.name} OPENED after {self._fails} failures - "
                       f"pausing for {CB_RESET_SECONDS}s")
                print(msg)
                diag.log.error(f"[breaker] {self.name} OPENED after {self._fails} "
                               f"consecutive failures - pausing {CB_RESET_SECONDS}s "
                               f"(this lane is now DOWN until a probe succeeds)")

    def status(self) -> str:
        with self._lock:
            if self._fails < CB_FAILURE_THRESHOLD:
                return "closed"
            return "half-open" if time.time() - self._opened > CB_RESET_SECONDS else "open"


_CB_VISION = _CircuitBreaker("vision-gemma")
_CB_REASONING = _CircuitBreaker("reasoning-qwen")
_CB_DIVERSITY = _CircuitBreaker("diversity-llama")


# Eval instrumentation (no effect in normal operation; used by the bench eval card).
# Token ledger: thread-safe accumulator of usage reported by each endpoint, so the bench
# can report tokens/question for a multi-node swarm run without per-call plumbing.
_TOKEN_LEDGER: dict[str, int] = {"prompt": 0, "completion": 0, "calls": 0}
_TOKEN_LOCK = threading.Lock()

# Forced-down lanes: lane keys in this set are treated as offline (primary skipped,
# graceful-degradation fallback fires). Drives the eval card's robustness row.
_FORCED_DOWN_LANES: set[str] = set()


def reset_token_ledger() -> None:
    with _TOKEN_LOCK:
        _TOKEN_LEDGER.update(prompt=0, completion=0, calls=0)


def read_token_ledger() -> dict[str, int]:
    with _TOKEN_LOCK:
        return dict(_TOKEN_LEDGER)


def _ledger_add(prompt_tokens: int, completion_tokens: int) -> None:
    with _TOKEN_LOCK:
        _TOKEN_LEDGER["prompt"] += int(prompt_tokens or 0)
        _TOKEN_LEDGER["completion"] += int(completion_tokens or 0)
        _TOKEN_LEDGER["calls"] += 1


def set_forced_down_lanes(lanes) -> None:
    """Eval hook: mark lanes ('vision'|'reasoning'|'diversity') offline. Empty clears it."""
    global _FORCED_DOWN_LANES
    _FORCED_DOWN_LANES = set(lanes or ())


def circuit_status() -> dict:
    """Return health summary of both LLM endpoints."""
    return {
        "vision": {"model": get_vision_llm_config()["model"], "circuit": _CB_VISION.status()},
        "reasoning": {"model": get_reasoning_llm_config()["model"], "circuit": _CB_REASONING.status()},
        "diversity": {"model": get_diversity_llm_config()["model"], "circuit": _CB_DIVERSITY.status()},
    }


# Think-token stripping

def _strip_thinking(raw: str, reasoning: str = "") -> str:
    """Strip <think>...</think> and Gemma <channel|> tokens from both models."""
    content = raw.strip()
    if "<channel|>" in content:
        content = content.split("<channel|>")[-1].strip()
    content = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL).strip()
    if not content:
        m = _re.search(r"<think>([\s\S]*?)(?:</think>|$)", raw)
        content = m.group(1).strip() if m else raw.strip()
    if not content and reasoning:
        content = reasoning.strip()
    return content


# Low-level request functions

def _is_ollama_thinking_model(cfg: dict) -> bool:
    """True for Ollama reasoning models that use chain-of-thought by default (Qwen3, DeepSeek-R1). These require the native /api/chat endpoint to control thinking."""
    if cfg.get("supports_thinking_param"):
        return False
    if "/v1/" not in cfg.get("base_url", ""):
        return False
    model_l = cfg.get("model", "").lower()
    return ("qwen3" in model_l) or ("deepseek-r1" in model_l)


def _call_ollama_native(cfg: dict, system_prompt: str, user_prompt: str,
                        max_tokens: int, temperature: float,
                        enable_thinking: bool = False) -> str:
    """Ollama native /api/chat endpoint. Required for Qwen3 to control chain-of-thought; the OpenAI-compat endpoint ignores think-control."""
    base = cfg["base_url"].replace("/v1/chat/completions", "/api/chat")
    # When thinking is on Qwen3 spends a large token budget on hidden reasoning before
    # emitting the answer; too small a num_predict returns thinking-only / empty content.
    num_predict = max(max_tokens, 2048) if enable_thinking else max_tokens
    payload = {
        "model": cfg["model"],
        "stream": False,
        "think": enable_thinking,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(base, json=payload, headers=headers, timeout=MAX_LLM_CALL_SECONDS)
    resp.raise_for_status()
    data = resp.json()
    try:  # ollama native reports token counts as prompt_eval_count / eval_count
        _ledger_add(data.get("prompt_eval_count", 0), data.get("eval_count", 0))
    except Exception:
        pass
    msg = data.get("message", {})
    return _strip_thinking(msg.get("content") or "", msg.get("thinking") or "")


def _call_openai_compat(cfg: dict, system_prompt: str, user_prompt: str,
                         max_tokens: int, enable_thinking: bool, temperature: float) -> str:
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # llama-server (Jetson) has only one model loaded - sending a model name causes
    # HTTP 400 if the name doesn't exactly match what the server was started with.
    # Omit the field; llama-server uses its loaded model by default.
    if not cfg.get("supports_thinking_param"):
        payload["model"] = cfg["model"]
    # Gemma 4 on llama-server always thinks before answering - ~270-310 tokens of hidden
    # reasoning before any visible content appears. Any budget below 400 returns empty.
    if cfg.get("supports_thinking_param"):
        payload["max_tokens"] = max(max_tokens, 400)
    # Only send enable_thinking when explicitly enabling it - omitting the field is
    # safer than passing false, since older llama-server builds reject unknown fields.
    if cfg.get("supports_thinking_param") and enable_thinking:
        payload["enable_thinking"] = True
    headers = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    resp = requests.post(
        cfg["base_url"], json=payload, headers=headers,
        timeout=MAX_LLM_CALL_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()
    try:  # OpenAI-compat reports usage.prompt_tokens / completion_tokens
        u = data.get("usage") or {}
        _ledger_add(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
    except Exception:
        pass
    msg = data["choices"][0]["message"]
    raw = msg.get("content") or ""
    return _strip_thinking(raw, msg.get("reasoning_content") or "")


def _make_request(cfg: dict, system_prompt: str, user_prompt: str,
                  max_tokens: int, enable_thinking: bool, temperature: float) -> str:
    """Single request - raises on any failure, never returns an error string."""
    # Qwen3/DeepSeek-R1 on Ollama: native /api/chat with think:false (OpenAI endpoint
    # ignores think-control and returns empty content).
    if _is_ollama_thinking_model(cfg):
        return _call_ollama_native(cfg, system_prompt, user_prompt, max_tokens,
                                   temperature, enable_thinking)
    return _call_openai_compat(cfg, system_prompt, user_prompt, max_tokens, enable_thinking, temperature)


def _call_with_retry(cfg: dict, system_prompt: str, user_prompt: str,
                     max_tokens: int, enable_thinking: bool, temperature: float) -> str:
    """Retry with exponential backoff. Timeouts and 4xx errors abort immediately. 5xx and connection errors retry up to MAX_RETRIES."""
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return _make_request(cfg, system_prompt, user_prompt,
                                 max_tokens, enable_thinking, temperature)
        except requests.Timeout:
            return (f"[LLM timeout after {MAX_LLM_CALL_SECONDS}s "
                    f"- {cfg.get('model', '?')} @ {cfg.get('base_url', '?')}]")
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 0
            if code in (400, 401, 403):
                return f"[HTTP {code}: {exc}]"
            last_err = exc
        except (requests.ConnectionError, OSError) as exc:
            last_err = exc
        except Exception as exc:
            last_err = exc

        if attempt < MAX_RETRIES:
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"[LLM] {cfg.get('model', '?')} attempt {attempt + 1} failed "
                  f"({last_err}) - retry in {delay:.1f}s")
            time.sleep(delay)

    return f"[{last_err}]"


# Main call_llm with dual routing + circuit breaker

def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 512,
             enable_thinking: bool = True, temperature: float = 0.8,
             use_fast: bool | None = None, lane: str | None = None) -> str:
    """Route to the appropriate model lane with circuit breaker and graceful fallback."""
    # Lane routing. An explicit `lane` pins an agent to a model family (distributed
    # swarm); otherwise fall back to the original max_tokens heuristic (used by the
    # routing / summary / reflection calls). Each lane falls back to a different model
    # if its own is down, so a dead node degrades gracefully instead of failing.
    if lane == "vision":
        primary_cfg, primary_cb, sem_key = get_vision_llm_config(), _CB_VISION, "vision"
        fallback_cfg, fallback_cb = get_reasoning_llm_config(), _CB_REASONING
    elif lane == "diversity":
        primary_cfg, primary_cb, sem_key = get_diversity_llm_config(), _CB_DIVERSITY, "diversity"
        fallback_cfg, fallback_cb = get_reasoning_llm_config(), _CB_REASONING
    elif lane == "reasoning":
        primary_cfg, primary_cb, sem_key = get_reasoning_llm_config(), _CB_REASONING, "reasoning"
        fallback_cfg, fallback_cb = get_vision_llm_config(), _CB_VISION
    else:
        if use_fast is None:
            use_fast = (max_tokens <= 200)
        if use_fast:
            primary_cfg, primary_cb, sem_key = get_vision_llm_config(), _CB_VISION, "vision"
            fallback_cfg, fallback_cb = get_reasoning_llm_config(), _CB_REASONING
        else:
            primary_cfg, primary_cb, sem_key = get_reasoning_llm_config(), _CB_REASONING, "reasoning"
            fallback_cfg, fallback_cb = get_vision_llm_config(), _CB_VISION

    def _try(cfg: dict, cb: _CircuitBreaker, role: str) -> str | None:
        """Attempt one endpoint. Returns None to signal caller should try fallback.
        role is "primary" | "fallback" - used only for diagnostics."""
        model = cfg.get("model", "?")
        if cb.is_open:
            print(f"[CB] {cb.name} open - skipping")
            diag.log.warning(f"[call] lane={sem_key} {role}={model} SKIP circuit-open")
            diag.event(f"{sem_key}/{role}_cb_open")
            return None
        # Resource guard (preventive): if this endpoint runs on THIS host and the
        # host is under memory pressure, divert to the fallback endpoint instead of
        # risking an OOM crash of the local Ollama/llama.cpp server.
        try:
            import resource_guard
            if resource_guard.should_divert(cfg.get("base_url", "")):
                snap = resource_guard.snapshot_str()
                print(f"[ResourceGuard] {model} is local & host under "
                      f"memory pressure ({snap}) - diverting to fallback")
                diag.log.warning(f"[call] lane={sem_key} {role}={model} DIVERT resource-guard ({snap})")
                diag.event(f"{sem_key}/{role}_diverted")
                return None
        except Exception:
            pass  # guard must never break the call path
        _t0 = time.time()
        result = _call_with_retry(cfg, system_prompt, user_prompt,
                                  max_tokens, enable_thinking, temperature)
        _dt = time.time() - _t0
        # Detect LLM error strings ('[...]' under 300 chars, no real content)
        is_err = (result.startswith("[") and "CLAIM:" not in result
                  and len(result) < 300)
        if is_err:
            cb.record_failure()
            print(f"[LLM] {role} {model} FAILED in {_dt:.1f}s: {result[:120]}")
            diag.log.error(f"[call] lane={sem_key} {role}={model} FAIL {_dt:.1f}s err={result[:200]!r}")
            diag.event(f"{sem_key}/{role}_err")
            return None
        cb.record_success()
        diag.log.info(f"[call] lane={sem_key} {role}={model} OK {_dt:.1f}s {len(result)}chars")
        diag.event(f"{sem_key}/{role}_ok")
        return result

    with _lane_sem(sem_key):
        diag.log.debug(f"[call] lane={sem_key} START primary={primary_cfg.get('model','?')} "
                       f"fallback={fallback_cfg.get('model','?')} maxtok={max_tokens} "
                       f"forced={sem_key in _FORCED_DOWN_LANES}")
        served_by = None
        if sem_key in _FORCED_DOWN_LANES:
            # Eval robustness row: this node is forced offline - skip primary, exercise
            # the real graceful-degradation fallback path instead.
            print(f"[Eval] lane '{sem_key}' forced offline - diverting to fallback")
            diag.log.warning(f"[call] lane={sem_key} FORCED-OFFLINE (eval robustness)")
            diag.event(f"{sem_key}/forced_down")
            result = None
        else:
            result = _try(primary_cfg, primary_cb, "primary")
            if result is not None:
                served_by = primary_cfg.get("model", "?")
        if result is None:
            print(f"[LLM] Primary ({primary_cfg.get('model', '?')}) unavailable "
                  f"- trying fallback ({fallback_cfg.get('model', '?')})")
            diag.log.warning(f"[call] lane={sem_key} FALLBACK {primary_cfg.get('model','?')}"
                             f" -> {fallback_cfg.get('model','?')}")
            result = _try(fallback_cfg, fallback_cb, "fallback")
            if result is not None:
                served_by = fallback_cfg.get("model", "?")
        diag.log.info(f"[call] lane={sem_key} DONE served_by={served_by or 'NONE'}")
        diag.event(f"{sem_key}/served:{served_by or 'none'}")

    return result or "[All LLM endpoints unavailable - check Jetson and Ollama]"


# Entity extraction

def extract_entities(topic: str) -> list[str]:
    prompt = (
        f"Extract the 5 most important named entities (people, technologies, "
        f"organizations, concepts) from this topic. Return ONLY a comma-separated "
        f"list, nothing else.\n\nTopic: {topic}"
    )
    raw = call_llm("You extract named entities precisely.", prompt,
                   max_tokens=80, enable_thinking=False)
    if raw.startswith("[") and "," not in raw[:20]:
        return []
    return [e.strip() for e in raw.split(",") if e.strip() and len(e.strip()) < 80][:8]
