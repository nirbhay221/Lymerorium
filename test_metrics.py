"""
Offline unit tests for bench/metrics.py - pure arithmetic, NO LLM/network.
Verifies the eval-card math against hand-computed values before trusting it on a
multi-hour cluster run.

Run:  python test_metrics.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench"))

import metrics as M  # noqa: E402

_FAILURES: list[str] = []


def check(label: str, got, want, tol: float = 1e-9) -> None:
    if isinstance(want, float):
        ok = abs(got - want) <= tol
    else:
        ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        print(f"         got={got!r}  want={want!r}")
        _FAILURES.append(label)


def test_accuracy() -> None:
    print("accuracy:")
    check("3/4", M.accuracy([True, True, False, True]), 0.75)
    check("empty -> 0", M.accuracy([]), 0.0)


def test_brier() -> None:
    print("brier_score:")
    # confident & right (1.0,1) and confident & wrong (1.0,0): (0 + 1)/2 = 0.5
    check("confident half-wrong", M.brier_score([(1.0, 1), (1.0, 0)]), 0.5)
    # perfectly calibrated extremes -> 0
    check("perfect", M.brier_score([(1.0, 1), (0.0, 0)]), 0.0)
    # 0.5 guesses -> 0.25 each
    check("coin flips", M.brier_score([(0.5, 1), (0.5, 0)]), 0.25)


def test_ece() -> None:
    print("expected_calibration_error:")
    # Perfectly calibrated: 0.0
    check("perfect -> 0", M.expected_calibration_error([(1.0, 1), (0.0, 0)], n_bins=10), 0.0)
    # All say 0.9 confidence, half right -> bin conf 0.9, acc 0.5, gap 0.4
    pairs = [(0.9, 1), (0.9, 0), (0.9, 1), (0.9, 0)]
    check("overconfident 0.9 bin, 50% acc -> 0.4",
          M.expected_calibration_error(pairs, n_bins=10), 0.4, tol=1e-9)
    check("empty -> 0", M.expected_calibration_error([]), 0.0)


def test_win_rate() -> None:
    print("win_rate:")
    base = [True,  True,  False, False, True]
    swrm = [True,  False, True,  False, True]
    wr = M.win_rate(base, swrm)
    check("n", wr["n"], 5)
    check("both_correct", wr["both_correct"], 2)      # items 0,4
    check("only_baseline (regressed)", wr["only_baseline"], 1)  # item 1
    check("only_swarm (rescued)", wr["only_swarm"], 1)          # item 2
    check("both_wrong", wr["both_wrong"], 1)          # item 3
    check("baseline_acc", wr["baseline_acc"], 0.6)
    check("swarm_acc", wr["swarm_acc"], 0.6)
    check("delta zero", wr["swarm_minus_baseline"], 0.0)


def test_win_rate_misaligned() -> None:
    print("win_rate misaligned -> raises:")
    try:
        M.win_rate([True], [True, False])
        check("should have raised", True, False)
    except ValueError:
        check("raised ValueError", True, True)


def test_mcnemar() -> None:
    print("mcnemar_exact_p:")
    check("no discordant -> 1.0", M.mcnemar_exact_p(0, 0), 1.0)
    # b=0,c=10: two-sided exact = 2 * 0.5^10 = 0.001953125
    check("10-0 split highly significant", M.mcnemar_exact_p(0, 10), 2 * (0.5 ** 10))
    # symmetric 5/5 -> p capped at 1.0
    check("5-5 split -> 1.0", M.mcnemar_exact_p(5, 5), 1.0)
    # direction-independent
    check("symmetry b<->c", M.mcnemar_exact_p(2, 8), M.mcnemar_exact_p(8, 2))


def test_cost() -> None:
    print("cost_summary:")
    c = M.cost_summary([10.0, 20.0, 30.0], [100, 200, 300])
    check("mean latency", c["mean_latency_s"], 20.0)
    check("total latency", c["total_latency_s"], 60.0)
    check("mean tokens", c["mean_tokens"], 200.0)
    check("total tokens", c["total_tokens"], 600)
    check("no tokens -> 0 mean", M.cost_summary([1.0], [])["mean_tokens"], 0.0)


def test_routing() -> None:
    print("routing_report:")
    pairs = [
        ("numeric", "numeric"),   # correct
        ("numeric", "open"),      # numeric missed -> recall hit
        ("choice", "choice"),     # correct
        ("choice", "choice"),     # correct
        ("open", "numeric"),      # open misrouted; numeric over-fires -> precision hit
    ]
    r = M.routing_report(pairs)
    check("overall accuracy 3/5", r["accuracy"], 0.6)
    # numeric: tp=1, fp=1 (the open->numeric), fn=1 (numeric->open) => prec .5 rec .5 f1 .5
    check("numeric precision", r["per_class"]["numeric"]["precision"], 0.5)
    check("numeric recall", r["per_class"]["numeric"]["recall"], 0.5)
    # choice: tp=2, fp=0, fn=0 -> perfect
    check("choice f1", r["per_class"]["choice"]["f1"], 1.0)
    check("empty -> 0", M.routing_report([])["accuracy"], 0.0)


def test_robustness() -> None:
    print("robustness_delta:")
    d = M.robustness_delta(0.80, 0.72)
    check("abs drop", d["abs_drop"], 0.08)
    check("retention", d["retention"], 0.9)
    check("zero full acc -> 0 retention", M.robustness_delta(0.0, 0.0)["retention"], 0.0)


def main() -> int:
    print("\n=== Offline eval-card metric tests ===\n")
    test_accuracy()
    test_brier()
    test_ece()
    test_win_rate()
    test_win_rate_misaligned()
    test_mcnemar()
    test_cost()
    test_routing()
    test_robustness()
    print()
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} check(s) -> {_FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
