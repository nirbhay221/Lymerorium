"""
Offline unit tests for the single-answer path - NO LLM, NO network, NO Ollama/Jetson.
Pure regex + aggregation, so you can sanity-check detection and vote parsing in <1s
before spending ~10 min/question on a full bench run.

Covers:
  oracle_chat._answer_mode      - numeric / choice / open routing
  simulation._vote_single_answer - FINAL: extraction, plain majority, and
                                   confidence-weighted aggregation

Run:  python test_single_answer.py
Exits non-zero if any assertion fails.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "swarm_core"))

from oracle_chat import _answer_mode  # noqa: E402
from simulation import _vote_single_answer, VOTE_CONFIDENCE_WEIGHTED  # noqa: E402

_FAILURES: list[str] = []


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        print(f"         got={got!r}  want={want!r}")
        _FAILURES.append(label)


def _msg(agent: str, content: str, rnd: int = 1, conf: int = 50, error: bool = False) -> dict:
    return {"agent": agent, "content": content, "round": rnd, "confidence": conf, "error": error}


# _answer_mode

def test_answer_mode() -> None:
    print("_answer_mode:")

    # choice - needs the literal "Options:" plus >=2 lettered option lines
    choice_q = "Which is healthiest?\nOptions:\nA. Apple\nB. Soda\nC. Candy"
    check("choice block -> choice", _answer_mode(choice_q), "choice")

    # numeric - a numeric trigger phrase AND at least one digit present
    check("'how many' + digit -> numeric",
          _answer_mode("How many apples are left if I start with 10 and eat 3?"), "numeric")
    check("'calculate' + digit -> numeric", _answer_mode("Calculate 5 * 4"), "numeric")
    check("'solve' + digit -> numeric", _answer_mode("Solve for x: 2x = 8"), "numeric")
    check("'what is the sum' + digit -> numeric",
          _answer_mode("What is the sum of 12 and 30?"), "numeric")

    # open - no numeric trigger, no options block
    check("factual question -> open", _answer_mode("What is the capital of France?"), "open")
    # 'how many' WITHOUT any digit falls through to open (documents the digit requirement)
    check("'how many' no digit -> open", _answer_mode("How many states are there?"), "open")
    # a lettered list WITHOUT the literal 'Options:' is not a choice block
    check("lettered list, no 'Options:' -> open",
          _answer_mode("Pick one:\nA. left\nB. right"), "open")


# _vote_single_answer

def test_vote_numeric_plain() -> None:
    print("_vote_single_answer - numeric, plain count (weighted=False):")
    msgs = [
        _msg("Skeptic", "reasoning...\nFINAL: 42"),
        _msg("Realist", "reasoning...\nFINAL: 42"),
        _msg("Economist", "reasoning...\nFINAL: 7"),
    ]
    voted, tally = _vote_single_answer(msgs, "numeric", weighted=False)
    check("majority winner", voted, "42")
    check("tally is raw counts", tally, {"42": 2, "7": 1})


def test_vote_numeric_confidence_flips_majority() -> None:
    print("_vote_single_answer - numeric, confidence overrides a raw majority:")
    # Two low-confidence agents say 7; one high-confidence agent says 42.
    msgs = [
        _msg("Skeptic", "FINAL: 7", conf=10),
        _msg("Realist", "FINAL: 7", conf=10),
        _msg("Economist", "FINAL: 42", conf=95),
    ]
    plain, _ = _vote_single_answer(msgs, "numeric", weighted=False)
    check("plain count picks the 2-vote bloc", plain, "7")
    weighted, tally = _vote_single_answer(msgs, "numeric", weighted=True)
    check("confidence-weighted picks the confident agent", weighted, "42")
    check("weighted tally sums confidences", tally, {"7": 0.2, "42": 0.95})


def test_vote_choice() -> None:
    print("_vote_single_answer - choice letters:")
    msgs = [
        _msg("Skeptic", "I think...\nFINAL: B"),
        _msg("Realist", "Agreed.\nFINAL: B"),
        _msg("Visionary", "Actually...\nFINAL: C"),
    ]
    voted, _ = _vote_single_answer(msgs, "choice", weighted=False)
    check("choice majority", voted, "B")


def test_vote_tail_fallback() -> None:
    print("_vote_single_answer - tail fallback when no FINAL: line:")
    msgs = [
        _msg("Skeptic", "After working it through, the answer is 12"),
        _msg("Realist", "FINAL: 12"),
    ]
    voted, _ = _vote_single_answer(msgs, "numeric", weighted=False)
    check("last number in tail is counted", voted, "12")


def test_vote_last_message_per_agent() -> None:
    print("_vote_single_answer - only an agent's LAST message counts:")
    msgs = [
        _msg("Skeptic", "FINAL: 5", rnd=1),
        _msg("Skeptic", "FINAL: 9", rnd=2),   # revised position - this one wins
        _msg("Realist", "FINAL: 9", rnd=2),
    ]
    voted, tally = _vote_single_answer(msgs, "numeric", weighted=False)
    check("revised answer used", voted, "9")
    check("stale round-1 vote dropped", tally, {"9": 2})


def test_vote_excludes_non_debaters() -> None:
    print("_vote_single_answer - Synthesizer/SYSTEM excluded:")
    msgs = [
        _msg("Synthesizer", "VERDICT: 100\nFINAL: 100"),
        _msg("SYSTEM", "FINAL: 100"),
        _msg("Skeptic", "FINAL: 3"),
        _msg("Realist", "FINAL: 3"),
    ]
    voted, tally = _vote_single_answer(msgs, "numeric", weighted=False)
    check("Synthesizer/SYSTEM not voting", voted, "3")
    check("only debater votes tallied", tally, {"3": 2})


def test_vote_all_equal_confidence_matches_plain() -> None:
    print("_vote_single_answer - uniform confidence means weighted == plain winner:")
    msgs = [
        _msg("Skeptic", "FINAL: 8", conf=50),
        _msg("Realist", "FINAL: 8", conf=50),
        _msg("Economist", "FINAL: 1", conf=50),
    ]
    w_voted, _ = _vote_single_answer(msgs, "numeric", weighted=True)
    p_voted, _ = _vote_single_answer(msgs, "numeric", weighted=False)
    check("weighted agrees with plain under equal confidence", w_voted, p_voted)
    check("winner is the bloc", w_voted, "8")


def test_vote_confidence_clamped() -> None:
    print("_vote_single_answer - confidence clamped to [0,100]:")
    msgs = [_msg("Skeptic", "FINAL: 5", conf=150)]   # 150 -> clamp 100 -> weight 1.0
    _, tally = _vote_single_answer(msgs, "numeric", weighted=True)
    check("over-100 confidence clamps to weight 1.0", tally, {"5": 1.0})


def test_vote_no_parseable_votes() -> None:
    print("_vote_single_answer - nothing parseable:")
    msgs = [
        _msg("Skeptic", "I have no idea, honestly."),
        _msg("Realist", "Let's discuss the philosophy instead."),
    ]
    voted, tally = _vote_single_answer(msgs, "choice", weighted=True)
    check("empty answer", voted, "")
    check("empty tally", tally, {})


def test_vote_default_uses_module_flag() -> None:
    print("_vote_single_answer - weighted=None honours module default:")
    # Construct a case whose winner differs between weighted and plain, then assert the
    # default path matches whichever mode VOTE_CONFIDENCE_WEIGHTED selects.
    msgs = [
        _msg("Skeptic", "FINAL: 7", conf=10),
        _msg("Realist", "FINAL: 7", conf=10),
        _msg("Economist", "FINAL: 42", conf=95),
    ]
    default_voted, _ = _vote_single_answer(msgs, "numeric")  # weighted=None
    expected = "42" if VOTE_CONFIDENCE_WEIGHTED else "7"
    check(f"default follows VOTE_CONFIDENCE_WEIGHTED={VOTE_CONFIDENCE_WEIGHTED}",
          default_voted, expected)


def main() -> int:
    print(f"\n=== Offline single-answer tests (VOTE_CONFIDENCE_WEIGHTED={VOTE_CONFIDENCE_WEIGHTED}) ===\n")
    test_answer_mode()
    test_vote_numeric_plain()
    test_vote_numeric_confidence_flips_majority()
    test_vote_choice()
    test_vote_tail_fallback()
    test_vote_last_message_per_agent()
    test_vote_excludes_non_debaters()
    test_vote_all_equal_confidence_matches_plain()
    test_vote_confidence_clamped()
    test_vote_no_parseable_votes()
    test_vote_default_uses_module_flag()

    print()
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} check(s) -> {_FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
