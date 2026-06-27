"""3-tier query routing: Synthesizer alone, selected agents, or full swarm based on query complexity."""


import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))

import agents as ag
import tools as t
from memory import MEMORY, AGENT_MEMORY

# Agent role descriptions (for the picker prompt)

_AGENT_ROLES = {
    "Skeptic": "challenges weak evidence and logical fallacies",
    "Visionary": "connects dots and paints future possibilities",
    "Realist": "anchors in current capabilities and constraints",
    "Ethicist": "examines societal impact, fairness, unintended consequences",
    "Technologist": "digs into implementation details and engineering effort",
    "Economist": "models incentives, cost structures, market forces",
    "Contrarian": "takes the least-popular defensible position",
    "Synthesizer": "finds underlying truth and builds integrated positions",
}

_SEVERITY_SYSTEM = """\
You are a query router. Rate the complexity of the user's question.
Reply with ONLY one word: simple, medium, or complex.

  simple   - factual, 1-2 sentence answer (definitions, current time, how-to basics)
  medium   - needs a few expert perspectives but has a reasonably clear answer
  complex  - genuinely uncertain, no clear right answer (default to this when unsure)"""

# Fast-path: short factual questions that never need multi-agent debate.
# Negative lookahead blocks advisory phrasing ("best", "better", "should") from
# being classified as simple - those need multi-agent debate.
_SIMPLE_RE = re.compile(
    r'^('
    r'what is (?!the best|a good|better|worse|right|wrong)'
    r'|what are (?!the best|good practices|better)'
    r'|what does|what do'
    r'|who is|who are'
    r'|when did|when is|when was'
    r'|where is|where are'
    r'|how many|how much'
    r'|how do i|how to'
    r'|what time|current time'
    r'|define |list |show me|explain |describe '
    r')',
    re.IGNORECASE,
)

# Fast-path: ethical / policy / existential topics that always warrant full debate
_COMPLEX_RE = re.compile(
    r'\b('
    r'should (?:we|governments?|humans?|society|AI|companies|people)'
    r'|ought to'
    r'|will (?:AI|AGI|robots?|automation|technology).{0,40}(?:humanity|society|humans?|jobs?|world|future)'
    r'|will humans?.{0,40}(?:survive|extinct|die)'
    r'|(?:AI|AGI|robot).{0,30}(?:rights?|consciousness|sentien|personhood)'
    r'|existential (?:risk|threat|crisis)'
    r'|(?:regulate|ban|control) (?:AI|AGI|technology)'
    r'|moral (?:dilemma|question|implications?)'
    r'|ethics? of'
    r'|meaning of (?:life|existence)'
    r'|end of (?:humanity|civilization|work|jobs?)'
    r'|future of (?:humanity|work|society|democracy|civilization)'
    r'|(?:good|bad|right|wrong) for (?:society|humanity|the world)'
    r'|geopolit'
    r'|nuclear war'
    r'|climate (?:change|crisis).{0,30}(?:should|policy|must)'
    r')',
    re.IGNORECASE,
)

_AGENT_PICKER_SYSTEM = f"""\
You are an orchestrator. Given a query, pick the 2-3 most relevant agents.
Available agents and what they do:
{chr(10).join(f'  {name}: {role}' for name, role in _AGENT_ROLES.items() if name != "Synthesizer")}

Reply with ONLY a comma-separated list of agent names, nothing else.
Example: Technologist, Skeptic, Economist"""

_DEBATE_PICKER_SYSTEM = """\
You are an orchestrator selecting debate agents for a specific topic.
Pick the 5-6 most relevant agents from this list:
  Skeptic:      challenges weak evidence and logical fallacies
  Visionary:    connects dots and paints future possibilities
  Realist:      anchors in current capabilities and constraints
  Ethicist:     examines societal impact, fairness, unintended consequences
  Technologist: digs into implementation details and engineering effort
  Economist:    models incentives, cost structures, market forces
  Contrarian:   takes the least-popular defensible position

Always include Contrarian. Reply with ONLY a comma-separated list, nothing else.
Example: Skeptic, Contrarian, Ethicist, Visionary, Technologist"""

_TOOL_PICKER_SYSTEM = """\
You are a routing agent. Given a query, decide which single tool best helps answer it.
Reply with ONLY one word:

  camera           - query is about what is physically visible right now
  web_fetch        - query needs a full article or deep research from the web
  web              - query needs recent news, quick facts, or external snippets
  knowledge_graph  - query is about topics already debated in this session
  filesystem       - query is about local files, project structure, or code
  time             - query needs the current date, time, or timezone
  none             - no tool needed"""


# Single-answer detection
# Tasks with one verifiable answer (a number, or a multiple-choice letter). The swarm
# DEGRADES these - adversarial debate talks agents off the correct answer and prose
# synthesis loses it (this is what crushed GSM8K: 90% solo -> 20% swarm). So numeric
# questions are answered by a single chain-of-thought pass; choice questions, where the
# swarm's truthfulness edge actually helps, stay on the swarm but in vote-backed mode.

_CHOICE_RE = re.compile(r'(?m)^\s*[A-H][\.\)]\s+\S')   # a lettered options block
_NUMERIC_RE = re.compile(
    r'\b(how many|how much|calculate|compute|solve|what is the '
    r'(?:sum|product|total|average|difference|result|value))\b',
    re.IGNORECASE,
)


def _answer_mode(query: str) -> str:
    if "Options:" in query and len(_CHOICE_RE.findall(query)) >= 2:
        return "choice"
    if _NUMERIC_RE.search(query) and re.search(r'\d', query):
        return "numeric"
    return "open"


def _numeric_solo(query: str) -> dict:
    """One careful chain-of-thought pass on the reasoning lane - no debate.
    Restores the single-model behaviour that scores ~90% on GSM8K."""
    raw = ag.call_llm(
        "You are a careful mathematician. Show your working.",
        f"Solve this problem step by step, then end with a line 'FINAL: <number>'.\n\n"
        f"Problem: {query}",
        max_tokens=2048, enable_thinking=True, temperature=0.0, lane="reasoning",
    )
    return {
        "answer": raw,
        "tier": "numeric_solo",
        "agents_used": ["Reasoner"],
        "tools_used": [],
        "source": "solo_cot",
        "topic": None,
    }


# Tier helpers

def _rate_severity(query: str) -> str:
    """
    Hybrid router - fast deterministic pass first, LLM only for ambiguous middle-ground.

    Tier 1 heuristic (microseconds):
      - Short factual question patterns: simple
      - Ethical/policy/existential keywords: complex

    Tier 2 fallback (LLM, only when heuristics don't decide):
      - Qwen3 classifies remaining ambiguous queries
    """
    q = query.strip()

    # Complex check runs FIRST - "what is the future of humanity?" must not hit simple
    if _COMPLEX_RE.search(q):
        return "complex"

    # Fast-path simple: short factual question (≤12 words + known opener)
    if len(q.split()) <= 12 and _SIMPLE_RE.match(q):
        return "simple"

    # LLM fallback for genuinely ambiguous cases (temperature=0.1 for stability)
    raw = ag.call_llm(_SEVERITY_SYSTEM, f"Query: {q}", max_tokens=10,
                      enable_thinking=False, temperature=0.1).strip().lower()
    for word in ("simple", "medium", "complex"):
        if word in raw:
            return word
    return "medium"


def _pick_debate_agents(query: str) -> list[str]:
    """
    Dynamic Role Assignment + AgentDropout.

    LLM selects the 5-6 most topic-relevant debate agents, then
    reputation.select_top_agents() ensures historically strong agents
    aren't dropped purely because of LLM randomness.
    Always includes Contrarian (structural group role).
    """
    import reputation as _rep

    all_debate = [n for n in _AGENT_ROLES if n != "Synthesizer"]

    raw = ag.call_llm(
        _DEBATE_PICKER_SYSTEM, f"Topic: {query}",
        max_tokens=80, enable_thinking=False, temperature=0.2,
    ).strip()

    names = [
        n.strip() for n in raw.split(",")
        if n.strip() in _AGENT_ROLES and n.strip() != "Synthesizer"
    ]
    if "Contrarian" not in names:
        names.append("Contrarian")

    if len(names) < 4:
        return all_debate  # LLM returned garbage - use all agents

    # AgentDropout retention: keep top-reputation agents within the LLM's selection
    selected = _rep.select_top_agents(names, n=min(6, len(names)))
    return selected if len(selected) >= 4 else all_debate


def _pick_agents(query: str) -> list[str]:
    raw = ag.call_llm(_AGENT_PICKER_SYSTEM, f"Query: {query}", max_tokens=60,
                      enable_thinking=False, temperature=0.3).strip()
    names = [n.strip() for n in raw.split(",") if n.strip() in _AGENT_ROLES]
    return names[:3] if names else ["Skeptic", "Realist", "Technologist"]


def _pick_tool(query: str) -> str:
    raw = ag.call_llm(_TOOL_PICKER_SYSTEM, f"Query: {query}", max_tokens=20,
                      enable_thinking=False, temperature=0.3).strip().lower()
    for word in ("camera", "web_fetch", "knowledge_graph", "filesystem", "time", "web", "none"):
        if word in raw:
            return word
    return "none"


def _run_tool(tool_choice: str, query: str, image_b64: str) -> str:
    if tool_choice == "camera":
        return t.camera_snapshot(query, image_b64)
    if tool_choice == "web_fetch":
        return t.web_search_and_fetch(query)
    if tool_choice == "web":
        return t.web_search_tool(query, max_results=5)
    if tool_choice == "knowledge_graph":
        return t.search_knowledge_graph(query)
    if tool_choice == "filesystem":
        return t.list_directory(".")
    if tool_choice == "time":
        return t.get_current_time("UTC")
    return ""


# Tier 1: Synthesizer with ALL relevant tools

def _gather_all_tools(query: str, image_b64: str) -> tuple[list[str], str]:
    """
    The chat-facing Synthesizer gets the fullest possible context:
    runs the model-picked primary tool + always queries the KG +
    always includes current time as a cheap anchor.
    Returns (tools_used, combined_block).
    """
    tools_used: list[str] = []
    parts: list[str] = []

    # 1. Always check the knowledge graph (cheap, no network)
    kg_result = t.search_knowledge_graph(query)
    if kg_result and "No matches" not in kg_result:
        tools_used.append("knowledge_graph")
        parts.append(f"[knowledge_graph]\n{kg_result[:600]}")

    # 2. Pick the strongest primary tool for the query
    primary = _pick_tool(query)
    if primary == "camera":
        out = t.camera_snapshot(query, image_b64)
        if out:
            tools_used.append("camera")
            parts.append(f"[camera]\n{out[:600]}")
    elif primary == "web_fetch":
        out = t.web_search_and_fetch(query)
        if out:
            tools_used.append("web_fetch")
            parts.append(f"[web_fetch]\n{out[:1400]}")
    elif primary == "web":
        out = t.web_search_tool(query, max_results=5)
        if out:
            tools_used.append("web")
            parts.append(f"[web]\n{out[:1200]}")
    elif primary == "filesystem":
        out = t.list_directory(".")
        if out:
            tools_used.append("filesystem")
            parts.append(f"[filesystem]\n{out[:600]}")

    # 3. Always include current time - cheap temporal anchor
    if "time" not in tools_used:
        tools_used.append("time")
        parts.append(f"[time]\n{t.get_current_time('UTC')}")

    return tools_used, "\n\n".join(parts)


def _tier1_simple(query: str, image_b64: str) -> dict:
    synth = ag.get_agent("Synthesizer")
    tools_used, tool_block = _gather_all_tools(query, image_b64)

    prompt = (
        f"User asked: \"{query}\"\n\n"
        f"Context from your tools:\n{tool_block}\n\n"
        f"Answer directly and concisely. Use the tool data where it's relevant."
    )
    reply = ag.call_llm(synth["personality"], prompt, max_tokens=1024, enable_thinking=False,
                        temperature=synth.get("temperature", 0.4))

    return {
        "answer": reply,
        "tier": "simple",
        "agents_used": ["Synthesizer"],
        "tools_used": tools_used,
        "source": "synthesizer",
        "topic": None,
    }


# Tier 2: Selected agents + Synthesizer

def _tier2_medium(query: str, image_b64: str) -> dict:
    chosen = _pick_agents(query)

    def _one_agent(name: str) -> str:
        agent = ag.get_agent(name)
        tool = _pick_tool(f"{query} from a {name.lower()} angle")
        tool_output = _run_tool(tool, query, image_b64)
        tool_block = f"\nRelevant information ({tool}):\n{tool_output}\n" if tool_output else ""
        prompt = f"""The user asks: "{query}"{tool_block}
As {name} ({_AGENT_ROLES[name]}), give your expert take in 2-3 sentences. Be specific."""
        response = ag.call_llm(agent["personality"], prompt, max_tokens=768, enable_thinking=False,
                               temperature=agent.get("temperature", 0.8))

        # Persist this agent's position to AGENT_MEMORY so mini-debates contribute to growth
        if response and len(response.strip()) > 20:
            claim = response.strip().split("\n")[0][:240]
            try:
                AGENT_MEMORY.save_position(name, query, claim)
            except Exception as exc:
                print(f"[Tier2] AGENT_MEMORY save failed for {name}: {exc}")

        return f"{name}: {response}"

    # Run agents sequentially (hardware-safe for single-GPU inference)
    opinions: list[str] = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = {pool.submit(_one_agent, n): n for n in chosen}
        for fut in as_completed(futures):
            try:
                opinions.append(fut.result())
            except Exception as exc:
                opinions.append(f"{futures[fut]}: [error: {exc}]")

    # Synthesizer combines
    synth = ag.get_agent("Synthesizer")
    combined = "\n\n".join(opinions)
    synth_prompt = f"""The user asked: "{query}"

Expert perspectives gathered:
{combined}

Synthesize these into a single clear, direct answer. Lead with the answer, then note key nuances."""
    final = ag.call_llm(synth["personality"], synth_prompt, max_tokens=1024, enable_thinking=False,
                        temperature=synth.get("temperature", 0.4))

    return {
        "answer": final,
        "tier": "medium",
        "agents_used": chosen + ["Synthesizer"],
        "tool_used": None,
        "source": "mini_debate",
        "agent_opinions": opinions,
        "topic": None,
    }


def _should_debate(query: str) -> bool:
    """
    iMAD: predict whether multi-agent debate will meaningfully
    improve over a single expert answer. Saves 10-20 min on queries that don't need it.

    Step 1 - single-agent self-critique: identify genuine uncertainty.
    Step 2 - debate-utility prediction: will multiple perspectives actually help?
    Defaults to True (debate anyway) if LLM is unavailable.
    """
    synth = ag.get_agent("Synthesizer")

    critique = ag.call_llm(
        synth["personality"],
        f"Question: {query}\n\n"
        f"In 2 sentences: (1) Is there a clear expert consensus or is this genuinely contested? "
        f"(2) What are the main sources of uncertainty or disagreement?",
        max_tokens=100, enable_thinking=False, temperature=0.3,
    )
    if critique.startswith("[") and len(critique) < 200:
        return True  # LLM error - default to debate

    raw = ag.call_llm(
        "You decide if multi-agent debate adds value over a single expert.",
        f"Question: {query}\n\nSelf-critique: {critique}\n\n"
        f"Will 6-8 specialized agents debating this provide meaningfully better insight "
        f"than one informed expert? Consider: genuine controversy, value trade-offs, "
        f"or deep technical uncertainty all favour debate. Clear factual answers do not.\n"
        f"Reply with ONLY: YES or NO",
        max_tokens=10, enable_thinking=False, temperature=0.1,
    ).strip().upper()

    should = "YES" in raw      # default NO if unclear - avoids burning 10-20 min on easy queries
    if not should:
        print(f"[iMAD] Debate skipped for: {query[:60]}")
    return should


# Tier 3: Full swarm - non-blocking
# Returns a Tier 2 answer immediately, fires full 8-agent swarm in background.
# When the swarm finishes, result lands in MEMORY - next related query gets it.

def _tier3_complex(query: str, image_b64: str, background_swarm, answer_mode: str = "open") -> dict:
    # iMAD: predict whether debate will actually help.
    # If not, drop to Tier 2 - saves 10-20 min on clear-enough questions.
    if not _should_debate(query):
        result = _tier2_medium(query, image_b64)
        result["tier"] = "complex_skipped_imad"
        result["note"] = "Full debate skipped - single expert consensus sufficient."
        return result

    # Dynamic Role Assignment: select the best agents for
    # this specific topic before firing the full debate in the background.
    _debate_agents = _pick_debate_agents(query)
    n_agents = len(_debate_agents)
    print(f"[Tier3] Dynamic agent selection: {_debate_agents}")

    # Immediate answer from 3 agents while full debate runs in background
    preliminary = _tier2_medium(query, image_b64)
    preliminary["tier"] = "complex_preliminary"
    preliminary["note"] = (
        f"Full {n_agents}-agent debate is running in the background. "
        "Ask again soon for the complete verdict."
    )

    # DOWN: even when iMAD says "debate is useful", if the
    # preliminary multi-agent answer is already high-confidence, the full
    # 10-20 min swarm debate adds marginal value. Exit early and save compute.
    try:
        _conf_raw = ag.call_llm(
            "You rate answer confidence on a 0-100 scale.",
            f"Question: {query}\n\nAnswer: {(preliminary.get('answer') or '')[:400]}\n\n"
            f"How confident and complete is this answer? Consider factual certainty, "
            f"coverage of key perspectives, and absence of major gaps.\n"
            f"Reply with ONLY an integer 0-100.",
            max_tokens=10, enable_thinking=False, temperature=0.1,
        ).strip()
        import re as _re_down
        _conf_m = _re_down.search(r'\d+', _conf_raw)
        _conf_score = int(_conf_m.group()) if _conf_m else 50
    except Exception:
        _conf_score = 50  # assume medium confidence on any error

    if _conf_score >= 70:
        print(f"[DOWN] Confidence={_conf_score}/100 >= 70 - full debate skipped")
        preliminary["tier"] = "complex_skipped_down"
        preliminary["note"] = (
            f"Full debate skipped - preliminary answer already confident "
            f"({_conf_score}/100). DOWN."
        )
        return preliminary
    print(f"[DOWN] Confidence={_conf_score}/100 < 70 - firing full debate")

    def _run_full_swarm():
        if background_swarm is not None:
            background_swarm.pause()
        try:
            from api import run_simulation
            from memory import MEMORY
            from config import MAX_ROUNDS
            result = run_simulation(query, max_rounds=MAX_ROUNDS, image_b64=image_b64,
                                    agent_filter=_debate_agents, answer_mode=answer_mode)
            verdict = result.get("verdict", "")
            # Skip storing cache hits or failed debates - same rule as BackgroundSwarm
            if not result.get("cache_hit") and verdict and not verdict.startswith("VERDICT: Debate could not complete"):
                MEMORY.add({
                    "topic": query,
                    "entities": result.get("entities", []),
                    "verdict": verdict,
                    "convergence_score": result.get("convergence_score", 0.0),
                })
                print(f"[Tier3] Full debate complete, stored in memory: {query[:60]}")
            else:
                print(f"[Tier3] Skipping MEMORY.add (cache_hit={result.get('cache_hit')}, verdict_ok={bool(verdict)})")
        except Exception as e:
            print(f"[Tier3] Background debate error: {e}")
        finally:
            if background_swarm is not None:
                background_swarm.resume()

    threading.Thread(target=_run_full_swarm, daemon=True).start()
    return preliminary


# Main entry point

def answer(query: str, past_verdict: dict | None, image_b64: str = "",
           background_swarm=None) -> dict:
    """
    Route and answer a query.

    1. Background memory hit: instant answer from Synthesizer (no severity check needed)
    2. No memory: rate severity, then tier 1 / 2 / 3
    """
    synth = ag.get_agent("Synthesizer")

    # Single-answer routing: a numeric task never goes to the swarm (it degrades it) -
    # answer it with one chain-of-thought pass. choice/open continue through normal routing.
    mode = _answer_mode(query)
    if mode == "numeric":
        return _numeric_solo(query)

    # Memory hit: answer from stored verdict
    if past_verdict and past_verdict.get("verdict"):
        prompt = f"""The user asked: "{query}"

A swarm of 8 AI agents already debated a related topic: "{past_verdict['topic']}"

Their verdict:
{past_verdict['verdict']}

Answer the user's question directly using this verdict. Flag any gaps."""
        reply = ag.call_llm(synth["personality"], prompt, max_tokens=1024, enable_thinking=False)
        return {
            "answer": reply,
            "tier": "memory",
            "agents_used": ["Synthesizer"],
            "tool_used": None,
            "source": "background_memory",
            "topic": past_verdict["topic"],
        }

    # Consolidated memory: distilled multi-debate insights (threshold 0.75)
    # Sits between raw-verdict hit and fresh routing - stricter match required.
    # Only exists once background swarm has debated 10+ high-quality topics.
    try:
        from consolidation import CONSOLIDATED_MEMORY
        consolidated = CONSOLIDATED_MEMORY.search(query, threshold=0.75)
    except Exception:
        consolidated = None
    if consolidated and consolidated.get("insight"):
        n_src = consolidated.get("source_count", "?")
        prompt = (
            f"The user asked: \"{query}\"\n\n"
            f"A distilled insight from {n_src} past swarm debates on related topics:\n"
            f"{consolidated['insight']}\n\n"
            f"Related debate topics: {', '.join(consolidated.get('topics', [])[:3])}\n\n"
            f"Answer the user's question using this distilled knowledge. "
            f"Be specific; flag if the insight only partially covers their question."
        )
        reply = ag.call_llm(synth["personality"], prompt, max_tokens=1024,
                            enable_thinking=False)
        return {
            "answer": reply,
            "tier": "consolidated_memory",
            "agents_used": ["Synthesizer"],
            "tool_used": None,
            "source": "consolidated_insight",
            "source_count": n_src,
            "topic": consolidated.get("topics", [None])[0],
        }

    # No memory: route by severity
    severity = _rate_severity(query)

    if severity == "simple":
        return _tier1_simple(query, image_b64)

    if severity == "medium":
        return _tier2_medium(query, image_b64)

    # complex - full swarm
    if background_swarm is not None:
        return _tier3_complex(query, image_b64, background_swarm, answer_mode=mode)

    # fallback if no background_swarm ref passed
    return _tier2_medium(query, image_b64)
