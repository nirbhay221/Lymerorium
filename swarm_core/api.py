"""Clean public interface for SwarmCore - called by app.py."""


import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from simulation import GRAPH  # noqa: E402
from agents import extract_entities, get_agent, call_llm  # noqa: E402
import knowledge_graph as kg  # noqa: E402
from config import MAX_ROUNDS  # noqa: E402
from memory import MEMORY  # noqa: E402


def _validate_cache(topic: str) -> dict | None:
    """Check VectorMemory for a valid cached verdict. Returns cached result or None."""
    past = MEMORY.search(topic, threshold=0.65)
    if not past:
        return None

    verdict_snippet = (past.get("verdict") or "")[:300]
    past_topic = past.get("topic", topic)
    age_hours = round((time.time() - past.get("timestamp", 0)) / 3600, 1)

    prompt = (
        f"A previous swarm debate covered: '{past_topic}'\n"
        f"Verdict summary: {verdict_snippet}\n"
        f"That debate was {age_hours:.1f} hours ago.\n\n"
        f"New topic: '{topic}'\n\n"
        f"Has anything materially changed that would significantly alter that verdict?\n"
        f"Reply with ONLY: YES or NO"
    )

    checkers = ["Skeptic", "Realist", "Contrarian"]
    yes_votes = 0
    error_votes = 0
    for name in checkers:
        agent = get_agent(name)
        reply = call_llm(
            agent["personality"], prompt,
            max_tokens=20, enable_thinking=False, temperature=0.2,
        ).strip().upper()
        # call_llm returns "[ExceptionStr]" on error - don't count as NO
        if reply.startswith("[") and not reply.startswith("[YES") and not reply.startswith("[NO"):
            error_votes += 1
        elif reply.startswith("YES"):
            yes_votes += 1

    print(f"[Cache] '{topic[:50]}' - {yes_votes}/3 changed, {error_votes}/3 errored "
          f"(past: '{past_topic[:40]}', {age_hours:.1f}h ago)")

    # If all validators errored (LLM down), don't pretend they said "no change"
    if error_votes == len(checkers):
        print(f"[Cache] All validators errored - running fresh debate to be safe")
        return None

    if yes_votes <= 1:
        # Secondary integrity check: verify the stored verdict is internally coherent before returning it.
        integrity_prompt = (
            f"Stored verdict for topic '{past_topic}':\n{verdict_snippet}\n\n"
            f"Does this verdict contain factual errors, internal contradictions, "
            f"or reasoning that contradicts well-established knowledge?\n"
            f"Reply with ONLY: SOUND or DOUBT"
        )
        integrity_checkers = ["Skeptic", "Ethicist"]
        doubt_votes = 0
        integrity_errors = 0
        for name in integrity_checkers:
            agent = get_agent(name)
            reply = call_llm(
                agent["personality"], integrity_prompt,
                max_tokens=10, enable_thinking=False, temperature=0.1,
            ).strip().upper()
            if reply.startswith("[") and "DOUBT" not in reply and "SOUND" not in reply:
                integrity_errors += 1
            elif "DOUBT" in reply:
                doubt_votes += 1

        print(f"[EquiMem] Integrity check - {doubt_votes}/2 doubt, "
              f"{integrity_errors}/2 errored")

        if doubt_votes >= 1 and integrity_errors < len(integrity_checkers):
            print(f"[EquiMem] Integrity suspect - discarding cached verdict, running fresh debate")
            return None

        return {
            "topic": topic,
            "entities": past.get("entities", []),
            "messages": [],
            "verdict": past.get("verdict", ""),
            "report_path": "",
            "rounds_completed": 0,
            "convergence_score": past.get("convergence_score", 0.0),
            "pivot_count": 0,
            "cache_hit": True,
            "cache_age_hours": age_hours,
            "cache_validators": checkers,
            "cache_change_votes": yes_votes,
            "cache_validated": error_votes == 0,  # False if some validators errored
        }

    return None  # agents detected change - run a fresh debate


_GATE_SAMPLES = 3   # SC draws for pre-debate confidence gate


def _quick_vote(topic: str, answer_mode: str) -> tuple[str, float] | None:
    """SC-3 pre-debate gate for single-answer tasks. If 3 independent draws agree unanimously, skip the full debate. Returns (answer, confidence) or None."""
    import re as _re
    from collections import Counter

    if answer_mode == "numeric":
        prompt = (
            "Solve this math problem step by step. "
            "End your response with a line 'FINAL: <the single numeric answer, digits only>'.\n\n"
            f"Problem: {topic}"
        )
        max_tok, think = 1024, True
    else:  # choice
        prompt = (
            f"{topic}\n\n"
            "Reason briefly, then end with a line 'FINAL: <the single correct option letter>'."
        )
        max_tok, think = 512, False

    def _extract(text: str) -> str | None:
        fm = _re.search(r"FINAL:\s*(.+)", text, _re.IGNORECASE)
        segment = (fm.group(1) if fm else text[-80:]).strip()
        if answer_mode == "numeric":
            nums = _re.findall(r"-?\d[\d,]*\.?\d*", segment)
            if not nums:
                return None
            return nums[-1].replace(",", "").rstrip(".")
        cm = _re.search(r"\b([A-H])\b", segment.upper())
        return cm.group(1) if cm else None

    preds = []
    for _ in range(_GATE_SAMPLES):
        try:
            raw = call_llm(
                "You are a precise reasoning assistant.",
                prompt, max_tokens=max_tok, temperature=0.7,
                enable_thinking=think, lane="reasoning",
            )
            ans = _extract(raw)
            if ans is not None:
                preds.append(ans)
        except Exception as exc:
            print(f"[Gate] sample error: {exc}")

    if not preds:
        return None

    winner, cnt = Counter(preds).most_common(1)[0]
    if cnt == _GATE_SAMPLES:  # unanimous - safe to skip debate
        print(f"[Gate] Unanimous {cnt}/{_GATE_SAMPLES} for '{winner}' - skipping full debate")
        return winner, 1.0

    print(f"[Gate] Disagreement ({cnt}/{_GATE_SAMPLES} for '{winner}') - running full debate")
    return None


def run_simulation(topic: str, max_rounds: int = MAX_ROUNDS, image_b64: str = "",
                   cancel_event=None, agent_filter: list | None = None,
                   stream_sink: list | None = None, answer_mode: str = "open",
                   skip_cache: bool = False, disable_gate: bool = False) -> dict:
    """Run a full swarm simulation. Checks cache first, then optionally runs the pre-debate gate before launching the full debate."""
    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Cancelled")

    import diag
    import agents as _agents
    run_id = str(int(time.time() * 1000))
    _t_start = time.time()
    diag.reset_ledger()
    _agents.reset_token_ledger()

    def _log_end(status: str, **kw) -> None:
        dt = time.time() - _t_start
        extra = " ".join(f"{k}={v}" for k, v in kw.items())
        diag.log.info(f"[SWARM END] run={run_id} status={status} dur={dt:.1f}s {extra}")
        diag.log.info(f"[SWARM END] run={run_id} tokens={_agents.read_token_ledger()} "
                      f"per-lane={diag.read_ledger()}")
        diag.log.info("=" * 60)

    try:
        _circ = _agents.circuit_status()
    except Exception as _e:
        _circ = f"<unavailable: {_e}>"
    diag.log.info("=" * 60)
    diag.log.info(f"[SWARM START] run={run_id} answer_mode={answer_mode} "
                  f"max_rounds={max_rounds} img={'yes' if image_b64 else 'no'} "
                  f"topic={topic[:80]!r}")
    diag.log.info(f"[SWARM START] run={run_id} circuits={_circ}")

    if not skip_cache:
        cached = _validate_cache(topic)
        if cached:
            _log_end("cache-hit")
            return cached

    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Cancelled")

    # Pre-debate gate: skip full debate when the primary model is already unanimous.
    # disable_gate=True forces full debate - used by eval's swarm_nogate arm.
    if answer_mode in ("numeric", "choice") and not disable_gate:
        gate = _quick_vote(topic, answer_mode)
        if gate is not None:
            voted_answer, gate_conf = gate
            conf_int = int(gate_conf * 100)
            _log_end("gate-hit", answer=voted_answer)
            return {
                "topic": topic,
                "entities": [],
                "messages": [],
                "verdict": (
                    f"VERDICT: {voted_answer}\n\n"
                    f"(pre-debate gate - SC {_GATE_SAMPLES}/{_GATE_SAMPLES} unanimous)\n\n"
                    f"FINAL: {voted_answer}\n\n"
                    f"CONFIDENCE_SCORE: {conf_int}"
                ),
                "report_path": "",
                "rounds_completed": 0,
                "convergence_score": gate_conf,
                "pivot_count": 0,
                "cache_hit": False,
                "gate_hit": True,
            }

    entities = extract_entities(topic)

    # Use consensus-free mode on topics where a previous debate failed to converge.
    use_free_mad = False
    if answer_mode == "open":
        contested_past = MEMORY.search(topic, threshold=0.55)
        if contested_past and contested_past.get("convergence_score", 1.0) < 0.45:
            use_free_mad = True
            print(f"[FREE-MAD] Activating - stored verdict convergence_score="
                  f"{contested_past['convergence_score']:.2f} < 0.45 (topic contested)")

    initial_state = {
        "topic": topic,
        "entities": entities,
        "round": 1,
        "max_rounds": max_rounds,
        "phase": "debate",
        "messages": [],
        "evidence_pool": {},
        "convergence_score": 0.0,
        "pivot_count": 0,
        "image_b64": image_b64,
        "verdict": "",
        "report_path": "",
        # GroupDebate / SELENE state - must be initialised here
        "group_summaries": {},
        "agent_last_claim": {},
        "agent_silent_count": {},
        # Dynamic agent selection
        "agent_filter": agent_filter or [],
        "last_pair_idx": 0,
        # MoA layered refinement
        "layer_context": "",
        # Anti-sycophancy
        "forced_challenge_done": False,
        # DRIFTJudge
        "drift_correction_done": False,
        # FREE-MAD
        "free_mad_mode": use_free_mad,
        "free_mad_scores": {},
        # Belief-Update Calibration
        "agent_confidence_history": {},
        "overconfidence_corrected": False,
        # DCI Epistemic Acts
        "unresolved_challenges": 0,
        # Single-answer mode ("open" | "numeric" | "choice")
        "answer_mode": answer_mode,
    }

    from simulation import _set_stream
    import resource_guard
    _set_stream(stream_sink)
    try:
        final_state = GRAPH.invoke(initial_state)
    except resource_guard.ResourceExhausted as e:
        # Host hit the critical memory line mid-debate - stop cleanly rather than
        # letting the local LLM OOM-crash. Returns a well-formed "aborted" result.
        print(f"[ResourceGuard] Debate aborted - {e}")
        _log_end("aborted", reason=repr(str(e)[:80]))
        return {
            "topic": topic,
            "entities": entities,
            "messages": [],
            "verdict": f"VERDICT: Debate aborted by resource guard - {e}",
            "report_path": "",
            "rounds_completed": 0,
            "convergence_score": 0.0,
            "pivot_count": 0,
            "cache_hit": False,
            "aborted": True,
        }
    finally:
        _set_stream(None)
    kg.soft_reset()

    verdict = final_state.get("verdict", "")

    # Store every completed debate in VectorMemory so cache validation works next time
    if verdict and not verdict.startswith("VERDICT: Debate could not complete"):
        MEMORY.add({
            "topic": topic,
            "entities": entities,
            "verdict": verdict,
            "convergence_score": final_state.get("convergence_score", 0.0),
            "timestamp": time.time(),
        })

    _clean_msgs = [m for m in final_state.get("messages", []) if not m.get("error")]
    _log_end("ok",
             rounds=final_state.get("round", max_rounds),
             convergence=round(final_state.get("convergence_score", 0.0), 2),
             msgs=len(_clean_msgs),
             verdict_len=len(verdict))
    return {
        "topic": topic,
        "entities": entities,
        "messages": _clean_msgs,
        "verdict": verdict,
        "report_path": final_state.get("report_path", ""),
        "rounds_completed": final_state.get("round", max_rounds),
        "convergence_score": final_state.get("convergence_score", 0.0),
        "pivot_count": final_state.get("pivot_count", 0),
        "cache_hit": False,
        "gate_hit": False,
        # Per-decision-point telemetry for post-hoc attribution (eval records these per row):
        "drift_fired": final_state.get("drift_correction_done", False),
        "forced_challenge_fired": final_state.get("forced_challenge_done", False),
        "free_mad_mode": final_state.get("free_mad_mode", False),
        "overconfidence_corrected": final_state.get("overconfidence_corrected", False),
    }
