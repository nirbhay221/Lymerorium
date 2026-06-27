"""Agent constitution learning: on very-low-confidence debates, propose and validate improvement rules per agent."""


import json
import threading
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent
_PATH = _BASE / "agent_constitutions.json"
_lock = threading.Lock()

_MAX_RULES_PER_AGENT = 5
_CONFIDENCE_TRIGGER = 30  # only activate on very poor debates

_PROPOSER_SYSTEM = """\
You are a meta-critic reviewing a failed multi-agent debate.
Read the weak verdict and agent arguments below.
Identify which single agent's contribution was most superficial, circular, or unhelpful.
Propose ONE specific, actionable improvement rule for that agent.

Reply in EXACTLY this format (no extra text):
AGENT: <agent name>
RULE: <one sentence starting with an action verb, e.g. "Always cite...", "Before claiming...">"""

_VALIDATOR_SYSTEM = """\
You are validating a proposed rule for a debate agent. Reply with ONLY: ACCEPT or REJECT.
ACCEPT if: rule is specific, actionable, and consistent with the agent's stated role.
REJECT if: rule is vague, contradicts the agent's core role, or is essentially a duplicate."""


# Persistence

def _load() -> dict:
    try:
        if _PATH.exists():
            return json.loads(_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


_constitutions: dict = _load()   # {agent_name: [rule_str, ...]}


def _save() -> None:
    try:
        tmp = _PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(_constitutions, indent=2), encoding="utf-8")
        tmp.replace(_PATH)
    except Exception as e:
        print(f"[Constitution] Save failed: {e}")


# Public API

def get_constitution(agent_name: str) -> list[str]:
    """Return active constitution rules for an agent (empty list if none)."""
    with _lock:
        return list(_constitutions.get(agent_name, []))


def get_all() -> dict:
    with _lock:
        return {k: list(v) for k, v in _constitutions.items()}


def maybe_update(verdict_text: str, messages: list[dict]) -> None:
    """
    Called after every debate from oracle_node.
    Only proceeds when CONFIDENCE_SCORE is very low - avoids touching good debates.
    Runs proposal + validation as a background job so oracle_node doesn't block.
    """
    import re, threading
    m = re.search(r"CONFIDENCE_SCORE:\s*(\d+)", verdict_text)
    if not m or int(m.group(1)) >= _CONFIDENCE_TRIGGER:
        return

    with _lock:
        total = sum(len(v) for v in _constitutions.values())
    if total >= _MAX_RULES_PER_AGENT * 7:
        print("[Constitution] Global rule cap reached - skipping")
        return

    # Run off the critical path so oracle_node returns immediately
    threading.Thread(
        target=_propose_and_validate,
        args=(verdict_text, list(messages)),
        daemon=True,
    ).start()


# Internal pipeline

def _propose_and_validate(verdict_text: str, messages: list[dict]) -> None:
    # Lazy imports to break circular dependency with agents.py
    from agents import call_llm, AGENT_INDEX

    debate_agents = [n for n in AGENT_INDEX if n != "Synthesizer"]

    agent_summary = "\n".join(
        f"{m['agent']}: {m['content'][:200]}"
        for m in messages[-14:]
        if not m.get("error") and m["agent"] in debate_agents
    )
    if not agent_summary:
        return

    proposal_prompt = (
        f"Bad debate verdict (very low confidence):\n{verdict_text[:400]}\n\n"
        f"Agent arguments:\n{agent_summary}\n\n"
        f"Which agent was weakest? Propose one improvement rule."
    )
    raw = call_llm(
        _PROPOSER_SYSTEM, proposal_prompt,
        max_tokens=80, enable_thinking=False, temperature=0.4,
    )

    agent_name = rule_text = None
    for line in raw.split("\n"):
        ls = line.strip()
        if ls.upper().startswith("AGENT:"):
            agent_name = ls[6:].strip()
        elif ls.upper().startswith("RULE:"):
            rule_text = ls[5:].strip()

    if not agent_name or not rule_text:
        return
    if agent_name not in AGENT_INDEX or agent_name == "Synthesizer":
        return
    if len(rule_text) < 10 or len(rule_text) > 200:
        return

    with _lock:
        current = _constitutions.get(agent_name, [])
        if len(current) >= _MAX_RULES_PER_AGENT:
            print(f"[Constitution] {agent_name} at cap ({_MAX_RULES_PER_AGENT}) - skipping")
            return
        if rule_text in current:
            return  # exact duplicate

    # Skeptic validates the rule (lazy import)
    agent_role = AGENT_INDEX[agent_name]["personality"][:150]
    val_prompt = (
        f"Agent: {agent_name}\n"
        f"Core role: {agent_role}...\n"
        f"Proposed rule: {rule_text}\n"
        f"Existing rules: {len(current)}/{_MAX_RULES_PER_AGENT}\n\n"
        f"ACCEPT or REJECT?"
    )
    skeptic_personality = AGENT_INDEX["Skeptic"]["personality"]
    verdict = call_llm(
        skeptic_personality, val_prompt,
        max_tokens=10, enable_thinking=False, temperature=0.1,
    ).strip().upper()

    if "ACCEPT" in verdict:
        with _lock:
            # Re-check cap here - another thread may have added a rule between
            # the first check and this write (two bad debates in quick succession).
            current_at_write = _constitutions.get(agent_name, [])
            if len(current_at_write) >= _MAX_RULES_PER_AGENT:
                return
            if rule_text in current_at_write:
                return
            _constitutions.setdefault(agent_name, []).append(rule_text)
            _save()
        print(f"[Constitution] Rule added to {agent_name}: {rule_text[:70]}")
    else:
        print(f"[Constitution] Rule rejected for {agent_name}: {rule_text[:70]}")
