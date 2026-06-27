"""
Pure metric math for the system eval card - NO LLM, NO network, NO project imports.
Everything here is deterministic arithmetic so it can be unit-tested offline (see
test_metrics.py) and trusted before a multi-hour run on the Jetson cluster.

Conventions:
  - A "pair" for calibration is (p, y): p = predicted confidence in [0,1], y in {0,1}.
  - Accuracy rows are booleans (correct / not).
  - Routing is evaluated as a multi-class confusion over modes {numeric, choice, open}.
"""

from __future__ import annotations

import math
from collections import Counter


# Accuracy

def accuracy(correct_flags: list[bool]) -> float:
    """Fraction correct in [0,1]. Empty -> 0.0."""
    if not correct_flags:
        return 0.0
    return sum(1 for c in correct_flags if c) / len(correct_flags)


# Calibration: Brier score + Expected Calibration Error

def brier_score(pairs: list[tuple[float, int]]) -> float:
    """
    Mean squared error between confidence and outcome: mean((p - y)^2). Lower = better
    calibrated. Range [0,1]. A model that says 100% and is right (or 0% and wrong)
    scores 0; confident-and-wrong scores 1.
    """
    if not pairs:
        return 0.0
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def expected_calibration_error(pairs: list[tuple[float, int]], n_bins: int = 10) -> float:
    """
    Expected Calibration Error: bin predictions by confidence, measure the gap between
    mean confidence and empirical accuracy per bin, weighted by bin population.
    """
    if not pairs:
        return 0.0
    n = len(pairs)
    # Bin edges over [0,1]; a confidence of exactly 1.0 falls in the last bin.
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in pairs:
        idx = min(n_bins - 1, max(0, int(p * n_bins)))
        bins[idx].append((p, y))
    ece = 0.0
    for b in bins:
        if not b:
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(y for _, y in b) / len(b)
        ece += (len(b) / n) * abs(conf - acc)
    return ece


# Paired arm comparison: win-rate + McNemar exact test

def win_rate(baseline_correct: list[bool], swarm_correct: list[bool]) -> dict:
    """
    Per-item breakdown of two arms run on the SAME items (must be aligned + equal length).
    Returns counts for the 2x2 contingency plus the swarm's net accuracy delta.
    """
    if len(baseline_correct) != len(swarm_correct):
        raise ValueError("arms must be aligned to the same items (equal length)")
    both = only_b = only_s = neither = 0
    for b, s in zip(baseline_correct, swarm_correct):
        if b and s:
            both += 1
        elif b and not s:
            only_b += 1
        elif s and not b:
            only_s += 1
        else:
            neither += 1
    n = len(baseline_correct)
    return {
        "n": n,
        "both_correct": both,
        "only_baseline": only_b,   # swarm regressed these
        "only_swarm": only_s,      # swarm rescued these
        "both_wrong": neither,
        "baseline_acc": (both + only_b) / n if n else 0.0,
        "swarm_acc": (both + only_s) / n if n else 0.0,
        "swarm_minus_baseline": (only_s - only_b) / n if n else 0.0,
    }


def mcnemar_exact_p(only_baseline: int, only_swarm: int) -> float:
    """
    Two-sided exact McNemar test on the discordant pairs (b = only_baseline-correct,
    c = only_swarm-correct). Uses the exact binomial (k successes of n=b+c at p=0.5) so it
    is valid for the small n a Jetson run produces - no scipy. Returns a p-value in (0,1].
    p < 0.05 means the accuracy difference between arms is unlikely to be noise.
    """
    n = only_baseline + only_swarm
    if n == 0:
        return 1.0
    k = min(only_baseline, only_swarm)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


# Cost

def cost_summary(latencies_s: list[float], token_counts: list[int]) -> dict:
    """Mean/total latency and tokens per question. Missing token data -> 0s."""
    n = len(latencies_s)
    lat_mean = sum(latencies_s) / n if n else 0.0
    tok_mean = sum(token_counts) / len(token_counts) if token_counts else 0.0
    return {
        "n": n,
        "mean_latency_s": round(lat_mean, 1),
        "total_latency_s": round(sum(latencies_s), 1),
        "mean_tokens": round(tok_mean, 1),
        "total_tokens": int(sum(token_counts)),
    }


# Routing quality (multi-class: numeric / choice / open)

def routing_report(pairs: list[tuple[str, str]]) -> dict:
    """
    pairs = (gold_mode, predicted_mode). Returns overall accuracy plus per-class
    precision / recall / F1 so a router that, say, over-fires "numeric" is visible.
    """
    if not pairs:
        return {"accuracy": 0.0, "n": 0, "per_class": {}}
    classes = sorted({g for g, _ in pairs} | {p for _, p in pairs})
    n = len(pairs)
    correct = sum(1 for g, p in pairs if g == p)
    per_class: dict[str, dict] = {}
    for c in classes:
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {"support": sum(1 for g, _ in pairs if g == c),
                        "precision": round(prec, 3), "recall": round(rec, 3),
                        "f1": round(f1, 3)}
    # Keys are "gold->pred" strings (not tuples) so the report is JSON-serializable.
    confusion = {f"{g}->{p}": c for (g, p), c in Counter((g, p) for g, p in pairs).items()}
    return {"accuracy": round(correct / n, 3), "n": n,
            "confusion": confusion,
            "per_class": per_class}


# Robustness

def robustness_delta(full_acc: float, degraded_acc: float) -> dict:
    """
    Graceful-degradation metric: accuracy with all nodes up vs with a node forced offline.
    `retention` is the fraction of full accuracy preserved when degraded (1.0 = no loss).
    """
    retention = (degraded_acc / full_acc) if full_acc > 0 else 0.0
    return {
        "full_acc": round(full_acc, 3),
        "degraded_acc": round(degraded_acc, 3),
        "abs_drop": round(full_acc - degraded_acc, 3),
        "retention": round(retention, 3),
    }
