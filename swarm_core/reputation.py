"""Agent reputation tracking: citation-based scoring, DyTopo pair effectiveness, and agent dropout retention."""


import json
import re
import random
import threading
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent
_PATH = _BASE / "agent_reputation.json"
_lock = threading.Lock()

_PAIR_COUNT = 3  # DyTopo rotates over 3 cross-pairs


# Persistence

def _defaults() -> dict:
    return {
        "agent_scores": {},   # {name: float}
        "agent_debates": {},  # {name: int}
        "pair_scores": [1.0] * _PAIR_COUNT,  # effectiveness per DyTopo pair
        "pair_uses": [0] * _PAIR_COUNT,
        "total_debates": 0,
    }


def _load() -> dict:
    try:
        if _PATH.exists():
            data = json.loads(_PATH.read_text(encoding="utf-8"))
            base = _defaults()
            for k, v in base.items():
                data.setdefault(k, v)
            return data
    except Exception:
        pass
    return _defaults()


_rep: dict = _load()


def _save() -> None:
    try:
        tmp = _PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(_rep, indent=2), encoding="utf-8")
        tmp.replace(_PATH)
    except Exception as e:
        print(f"[Reputation] Save failed: {e}")


# Core update

def record_verdict(
    verdict_text: str,
    participating_agents: list[str],
    last_pair_idx: int = 0,
) -> None:
    """
    Called after every completed debate.

    - Agents named in the verdict text receive citation credit (scaled by
      the verdict's confidence score).
    - Uncited agents decay slightly (Lamarckian branch of HiveMind).
    - The active DyTopo pair is credited when confidence >= 60.
    """
    with _lock:
        _rep["total_debates"] += 1

        m = re.search(r"CONFIDENCE_SCORE:\s*(\d+)", verdict_text)
        confidence = int(m.group(1)) if m else 50

        for name in participating_agents:
            _rep["agent_scores"].setdefault(name, 1.0)
            _rep["agent_debates"][name] = _rep["agent_debates"].get(name, 0) + 1
            if name in verdict_text:
                bonus = 1.0 * (confidence / 100)
                _rep["agent_scores"][name] += max(0.1, bonus)
            else:
                _rep["agent_scores"][name] = max(0.5, _rep["agent_scores"][name] * 0.97)

        # GoAgent pair effectiveness tracking
        if 0 <= last_pair_idx < _PAIR_COUNT:
            _rep["pair_uses"][last_pair_idx] += 1
            if confidence >= 60:
                _rep["pair_scores"][last_pair_idx] += 0.5
            else:
                _rep["pair_scores"][last_pair_idx] = max(
                    0.5, _rep["pair_scores"][last_pair_idx] * 0.98
                )

        _save()


# DyTopo pair selection

def get_agent_trust(agent_name: str) -> float:
    """
    Trust-Aware Sparse: normalised trust score for an agent in [0, 1].
    Based on accumulated HiveMind reputation - higher = more reliable across past debates.
    Returns 0.5 for agents with no history (cold-start neutral).
    """
    with _lock:
        scores = _rep["agent_scores"]
        if not scores:
            return 0.5
        score = scores.get(agent_name, 1.0)
        max_s = max(scores.values()) if scores else 1.0
        return min(1.0, score / max(max_s, 1.0))


def best_pair_idx(round_num: int) -> int:
    """
    GoAgent-style topology selection: weighted-random over pair effectiveness.
    Falls back to pure rotation when all scores are uniform (cold start).
    """
    with _lock:
        scores = list(_rep["pair_scores"])

    if max(scores) == min(scores):
        return (round_num - 1) % _PAIR_COUNT

    total = sum(scores)
    r = random.uniform(0, total)
    cumulative = 0.0
    for i, s in enumerate(scores):
        cumulative += s
        if r <= cumulative:
            return i
    return (round_num - 1) % _PAIR_COUNT


# Synthesizer prompt injection

def reputation_context(participating_agents: list[str]) -> str:
    """
    Returns a one-line hint for the Synthesizer listing the top-3 agents
    by historical debate contribution (Epistemic Context Learning).
    Empty string if the system has no history yet.
    """
    with _lock:
        scores = dict(_rep["agent_scores"])

    if not scores:
        return ""

    ranked = sorted(
        [n for n in participating_agents if n != "Synthesizer"],
        key=lambda n: scores.get(n, 1.0),
        reverse=True,
    )[:3]
    if not ranked:
        return ""
    items = ", ".join(f"{n} ({scores.get(n, 1.0):.1f})" for n in ranked)
    return (
        f"Agents with strongest track record across past debates: {items}. "
        f"Weight their arguments accordingly.\n"
    )


# AgentDropout-style retention

def select_top_agents(candidates: list[str], n: int = 5) -> list[str]:
    """
    From `candidates` (LLM-selected for topic relevance), retain top-n
    by reputation while always preserving Contrarian (structural role)
    and the highest-reputation agent.

    If there are fewer candidates than n, all are returned unchanged.
    """
    if len(candidates) <= n:
        return candidates

    with _lock:
        scores = dict(_rep["agent_scores"])

    must_keep: set[str] = set()
    if "Contrarian" in candidates:
        must_keep.add("Contrarian")
    top_rep = max(candidates, key=lambda a: scores.get(a, 1.0))
    must_keep.add(top_rep)

    rest = sorted(
        [a for a in candidates if a not in must_keep],
        key=lambda a: scores.get(a, 1.0),
        reverse=True,
    )
    return (list(must_keep) + rest)[: n]


# Status

def get_stats() -> dict:
    """Full reputation data for /swarm_status or debugging."""
    with _lock:
        return {
            "agent_scores":  dict(_rep["agent_scores"]),
            "agent_debates": dict(_rep["agent_debates"]),
            "pair_scores":   list(_rep["pair_scores"]),
            "pair_uses":     list(_rep["pair_uses"]),
            "total_debates": _rep["total_debates"],
        }
