"""
Swarm eval card: accuracy, calibration, cost, routing, robustness, and win-rate
across four arms (baseline, self_consistency, swarm, swarm_nogate) and four benchmarks.

Usage:
  python bench/eval_card.py --bench gsm8k,truthfulqa,strategyqa,mmlu --n 30
  python bench/eval_card.py --bench gsm8k --n 3 --arms baseline,swarm
  python bench/eval_card.py --no-robustness
  python bench/eval_card.py --resume bench/evalcard_20260624_004451.json
"""
import sys, os, re, json, time, random, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "swarm_core"))
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import requests
import config  # noqa: F401  (forces UTF-8 stdout + loads .env)
import agents as ag
from api import run_simulation
from oracle_chat import _answer_mode

import run_bench as RB       # reuse chat/extractors/loaders/gold
import metrics as M

HERE = os.path.dirname(os.path.abspath(__file__))
ROBUST_N = 5                 # items/bench for the node-down robustness sub-run
ROBUST_BENCHES = ("gsm8k", "truthfulqa")


# Confidence + token helpers

def _parse_conf(text: str, default: float = 0.5) -> float:
    """Pull a 0-100 confidence (swarm: CONFIDENCE_SCORE; baseline: CONFIDENCE) -> [0,1]."""
    m = re.search(r"CONFIDENCE(?:_SCORE)?:\s*(\d+)", text or "", re.IGNORECASE)
    if not m:
        return default
    return max(0.0, min(1.0, int(m.group(1)) / 100.0))


def _solo_chat(prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> tuple[str, int]:
    """Baseline single-model call that also returns token usage (for the cost metric)."""
    body = {"model": RB.QWEN_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "stream": False}
    r = requests.post(RB.QWEN_URL, json=body, timeout=300)
    r.raise_for_status()
    j = r.json()
    txt = RB.strip_think(j["choices"][0]["message"]["content"])
    u = j.get("usage") or {}
    tok = int(u.get("prompt_tokens", 0)) + int(u.get("completion_tokens", 0))
    return txt, tok


def _swarm_tokens_around(fn):
    """Run fn() with the token ledger zeroed; return (result, tokens_used)."""
    ag.reset_token_ledger()
    result = fn()
    led = ag.read_token_ledger()
    return result, led["prompt"] + led["completion"]


_SC_SAMPLES = 5   # self-consistency: independent draws per question

def _self_consistency(task) -> tuple:
    """Sample the primary model _SC_SAMPLES times, then take majority vote. Confidence = fraction agreeing with winner."""
    from collections import Counter
    if task["kind"] == "numeric":
        prompt = ("Solve this math word problem step by step. "
                  "End with the final number on its own line.\n\n"
                  f"Problem: {task['question']}")
        max_tok = 2048
    else:
        prompt = (task["route_query"] + " Reply with the letter and a one-sentence reason.")
        max_tok = 512
    preds, total_tok, raws = [], 0, []
    for _ in range(_SC_SAMPLES):
        raw, tok = _solo_chat(prompt, max_tokens=max_tok, temperature=0.7)
        raws.append(raw)
        total_tok += tok
        if task["kind"] == "numeric":
            preds.append(RB.extract_number(raw))
        else:
            preds.append(RB.extract_letter(raw, task["options_block"]))
    valid_str = [str(p) for p in preds if p is not None]
    if not valid_str:
        return None, 0.5, total_tok, "\n---\n".join(raws)
    winner, cnt = Counter(valid_str).most_common(1)[0]
    conf = cnt / len(valid_str)
    pred = (int(winner) if task["kind"] == "numeric" and winner.lstrip("-").isdigit()
            else winner)
    return pred, conf, total_tok, "\n---\n".join(raws)


# Bench loaders -> task dicts with a uniform shape:
#   {id, question, gold, kind, route_query, mode_gold, options_block?}

def load_gsm8k(n):
    out = []
    for t in RB.load_gsm8k(n):
        out.append({**t, "kind": "numeric", "route_query": t["question"],
                    "mode_gold": "numeric"})
    return out


def _choice_task(i, question, options_block, gold_letter):
    route_query = (f"Question: {question}\n\nOptions:\n{options_block}\n\n"
                   "Select the single most correct option.")
    return {"id": i, "question": question, "options_block": options_block,
            "gold": gold_letter, "kind": "choice", "route_query": route_query,
            "mode_gold": "choice"}


def load_truthfulqa(n):
    return [_choice_task(t["id"], t["question"], t["options_block"], t["gold"])
            for t in RB.load_truthfulqa(n)]


def load_strategyqa(n):
    path = os.path.join(HERE, "strategyqa.json")
    if not os.path.exists(path):
        raise FileNotFoundError("strategyqa.json missing - run: python bench/fetch_data.py")
    data = json.load(open(path, encoding="utf-8"))
    idx = random.Random(RB.SEED).sample(range(len(data)), min(n, len(data)))
    block = "A. Yes\nB. No"
    out = []
    for i in idx:
        d = data[i]
        gold = "A" if str(d["answer"]).lower() in ("yes", "true", "1") else "B"
        out.append(_choice_task(i, d["question"], block, gold))
    return out


def load_mmlu(n):
    path = os.path.join(HERE, "mmlu.json")
    if not os.path.exists(path):
        raise FileNotFoundError("mmlu.json missing - run: python bench/fetch_data.py")
    data = json.load(open(path, encoding="utf-8"))
    idx = random.Random(RB.SEED).sample(range(len(data)), min(n, len(data)))
    out = []
    for i in idx:
        d = data[i]
        letters = [chr(65 + k) for k in range(len(d["choices"]))]
        block = "\n".join(f"{letters[k]}. {c}" for k, c in enumerate(d["choices"]))
        gold = letters[int(d["answer"])]
        out.append(_choice_task(i, d["question"], block, gold))
    return out


LOADERS = {"gsm8k": load_gsm8k, "truthfulqa": load_truthfulqa,
           "strategyqa": load_strategyqa, "mmlu": load_mmlu}


# Arm execution -> (pred, conf, tokens, raw, meta)
# meta: per-question swarm telemetry (gate/cache/abort flags, rounds, mechanism fires). Empty for non-swarm arms.

SWARM_ARMS = ("swarm", "swarm_nogate")


def _swarm_meta(res: dict) -> dict:
    """Extract per-question decision-point telemetry from a run_simulation result."""
    return {
        "gate_hit":         res.get("gate_hit", False),          # debate SKIPPED by gate
        "cache_hit":        res.get("cache_hit", False),         # served from memory
        "aborted":          res.get("aborted", False),           # resource guard tripped
        "rounds":           res.get("rounds_completed", 0),      # debate rounds run
        "convergence":      round(res.get("convergence_score", 0.0), 2),
        "pivots":           res.get("pivot_count", 0),
        "drift_fired":      res.get("drift_fired", False),       # DRIFTJudge re-anchored
        "forced_challenge": res.get("forced_challenge_fired", False),  # anti-sycophancy
        "free_mad":         res.get("free_mad_mode", False),
        "overconf_fixed":   res.get("overconfidence_corrected", False),
    }


def run_item(task, arm):
    if task["kind"] == "numeric":
        if arm == "baseline":
            raw, tok = _solo_chat(
                "Solve this math word problem step by step. End with the final number on "
                "its own line, then a line 'CONFIDENCE: <0-100>' for how sure you are.\n\n"
                f"Problem: {task['question']}", max_tokens=2048)
            return RB.extract_number(raw), _parse_conf(raw), tok, raw, {}
        if arm == "self_consistency":
            pred, conf, tok, raw = _self_consistency(task)
            return pred, conf, tok, raw, {}
        # swarm / swarm_nogate
        res, tok = _swarm_tokens_around(
            lambda: run_simulation(task["question"], max_rounds=2, answer_mode="numeric",
                                   skip_cache=True, disable_gate=(arm == "swarm_nogate")))
        raw = res.get("verdict", "")
        return RB.extract_number(raw), _parse_conf(raw), tok, raw, _swarm_meta(res)

    # choice
    if arm == "baseline":
        raw, tok = _solo_chat(
            task["route_query"] + " Reply with the letter, a one-sentence reason, then a "
            "line 'CONFIDENCE: <0-100>'.", max_tokens=512)
        return RB.extract_letter(raw, task["options_block"]), _parse_conf(raw), tok, raw, {}
    if arm == "self_consistency":
        pred, conf, tok, raw = _self_consistency(task)
        return pred, conf, tok, raw, {}
    # swarm / swarm_nogate
    res, tok = _swarm_tokens_around(
        lambda: run_simulation(task["route_query"], max_rounds=2, answer_mode="choice",
                               skip_cache=True, disable_gate=(arm == "swarm_nogate")))
    raw = res.get("verdict", "")
    return RB.extract_letter(raw, task["options_block"]), _parse_conf(raw), tok, raw, _swarm_meta(res)


# Robustness sub-run: swarm with the Jetson (vision) lane forced offline

def run_robustness(tasks_by_bench, results):
    print("\n=== ROBUSTNESS: swarm with Jetson (vision) lane forced offline ===")
    out = {}
    ag.set_forced_down_lanes({"vision"})
    try:
        for bench, tasks in tasks_by_bench.items():
            if bench not in ROBUST_BENCHES:
                continue
            subset = tasks[:ROBUST_N]
            ids = {t["id"] for t in subset}
            # full-up accuracy on the SAME ids, taken from the main swarm rows
            full_flags = [r["correct"] for r in results["rows"]
                          if r["arm"] == "swarm" and r["bench"] == bench and r["id"] in ids]
            degraded = []
            for t in subset:
                try:
                    pred, _, _, _, _ = run_item(t, "swarm")
                    degraded.append(pred is not None and str(pred) == str(t["gold"]))
                except Exception as e:
                    print(f"  [robust {bench} {t['id']}] ERR {type(e).__name__}: {e}")
                    degraded.append(False)
            if full_flags:
                rd = M.robustness_delta(M.accuracy(full_flags), M.accuracy(degraded))
                out[bench] = rd
                print(f"  {bench}: full={rd['full_acc']:.2f} degraded={rd['degraded_acc']:.2f} "
                      f"retention={rd['retention']:.2f}")
    finally:
        ag.set_forced_down_lanes(set())   # always restore
    return out


# Resume helper

def _load_resume(path: str) -> tuple[dict, set, str]:
    """Load a partial card JSON and return (card, completed_set, out_path)."""
    with open(path, encoding="utf-8") as f:
        card = json.load(f)
    completed = {(r["bench"], r["arm"], r["id"]) for r in card["rows"]}
    n_done = len(completed)
    meta = card["meta"]
    print(f"[resume] {path}")
    print(f"[resume] {n_done} (bench, arm, id) combos done  "
          f"n={meta['n']}  arms={meta['arms']}  benches={meta['benches']}")
    return card, completed, path


# Metrics recompute from rows (used both fresh and on resume)

def _recompute_arm_metrics(card: dict, bench: str, arm: str) -> None:
    """Rebuild accuracy / calibration / cost for one (bench, arm) from card['rows']."""
    rows = [r for r in card["rows"] if r["bench"] == bench and r["arm"] == arm]
    if not rows:
        return
    flags      = [r["correct"]     for r in rows]
    lats       = [r["latency_s"]   for r in rows]
    toks       = [r["tokens"]      for r in rows]
    conf_pairs = [(r["conf"], int(r["correct"])) for r in rows]
    card["accuracy"].setdefault(bench, {})[arm] = round(M.accuracy(flags), 3)
    card["calibration"].setdefault(bench, {})[arm] = {
        "ece":   round(M.expected_calibration_error(conf_pairs), 3),
        "brier": round(M.brier_score(conf_pairs), 3),
    }
    card["cost"].setdefault(bench, {})[arm] = M.cost_summary(lats, toks)


def _recompute_winrates(card: dict, bench: str, arms: list[str]) -> None:
    """Rebuild per-arm win-rate for one bench from card['rows']."""
    by_id: dict[int, dict] = {}
    for r in card["rows"]:
        if r["bench"] == bench:
            by_id.setdefault(r["id"], {})[r["arm"]] = r["correct"]

    def _wr(arm_a: str, arm_b: str, key: str):
        aligned = [(v[arm_a], v[arm_b]) for v in by_id.values()
                   if arm_a in v and arm_b in v]
        if not aligned:
            return
        wr = M.win_rate([a for a, _ in aligned], [b for _, b in aligned])
        wr["mcnemar_p"] = round(
            M.mcnemar_exact_p(wr["only_baseline"], wr["only_swarm"]), 4)
        card["winrate"][key] = wr

    if "baseline" in arms and "swarm" in arms:
        _wr("baseline", "swarm", bench)
    if "self_consistency" in arms and "swarm" in arms:
        _wr("self_consistency", "swarm", f"{bench}_vs_sc")
    # Gate-isolation comparison: swarm (gate on) vs swarm_nogate (gate off).
    if "swarm" in arms and "swarm_nogate" in arms:
        _wr("swarm", "swarm_nogate", f"{bench}_gate_effect")


def _recompute_gate_analysis(card: dict, bench: str) -> None:
    """Split swarm arm accuracy into gated (debate skipped) vs debated (full debate)."""
    rows = [r for r in card["rows"] if r["bench"] == bench and r["arm"] == "swarm"]
    if not rows:
        return
    gated   = [r for r in rows if r.get("meta", {}).get("gate_hit")]
    debated = [r for r in rows if not r.get("meta", {}).get("gate_hit")]
    card["gate_analysis"][bench] = {
        "n_total":        len(rows),
        "n_gated":        len(gated),
        "n_debated":      len(debated),
        "gate_fire_rate": round(len(gated) / len(rows), 3),
        "acc_overall":    round(M.accuracy([r["correct"] for r in rows]), 3),
        "acc_gated":      round(M.accuracy([r["correct"] for r in gated]), 3) if gated else None,
        "acc_debated":    round(M.accuracy([r["correct"] for r in debated]), 3) if debated else None,
    }


# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench",   default="gsm8k,truthfulqa,strategyqa,mmlu")
    ap.add_argument("--n",       type=int, default=30)
    ap.add_argument("--arms",    default="baseline,swarm")
    ap.add_argument("--no-robustness", action="store_true")
    ap.add_argument("--resume",  metavar="FILE",
                    help="path to a partial evalcard JSON to continue from")
    args = ap.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────────
    if args.resume:
        card, completed, out_path = _load_resume(args.resume)
        n      = card["meta"]["n"]
        arms   = card["meta"]["arms"]
        benches = [b for b in card["meta"]["benches"] if b in LOADERS]
    else:
        completed = set()
        benches = [b for b in args.bench.split(",") if b in LOADERS]
        arms    = args.arms.split(",")
        n       = args.n
        stamp   = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(HERE, f"evalcard_{stamp}.json")
        card = {
            "meta": {"n": n, "arms": arms, "benches": benches,
                     "seed": RB.SEED, "started": stamp},
            "rows": [], "accuracy": {}, "calibration": {}, "cost": {},
            "routing": {}, "winrate": {}, "robustness": {}, "gate_analysis": {},
        }

    def save():
        json.dump(card, open(out_path, "w", encoding="utf-8"), indent=2)

    print(f"=== EVAL CARD  n={n}  arms={arms}  benches={benches} ===")
    if completed:
        print(f"    resuming - {len(completed)} rows already done\n")
    print(f"Output -> {out_path}\n")

    tasks_by_bench: dict[str, list] = {}

    # ── Main loop ──────────────────────────────────────────────────────────────
    for bench in benches:
        try:
            tasks = LOADERS[bench](n)
        except Exception as e:
            print(f"[skip {bench}] {type(e).__name__}: {e}")
            continue
        tasks_by_bench[bench] = tasks

        for arm in arms:
            n_new = 0
            for j, task in enumerate(tasks):
                key = (bench, arm, task["id"])
                if key in completed:
                    print(f"[{bench:10} {arm:8} {j+1:2}/{len(tasks)}] SKIP (already done)")
                    continue

                t0 = time.time()
                try:
                    pred, conf, tok, raw, meta = run_item(task, arm)
                    err = None
                except Exception as e:
                    pred, conf, tok, raw, meta, err = None, 0.0, 0, "", {}, f"{type(e).__name__}: {e}"
                dt = time.time() - t0

                ok = (pred is not None and str(pred) == str(task["gold"]))
                row = {
                    "bench": bench, "arm": arm, "id": task["id"],
                    "gold": task["gold"], "pred": pred, "correct": ok,
                    "conf": round(conf, 2), "latency_s": round(dt, 1),
                    "tokens": tok, "error": err,
                }
                if meta:                 # swarm arms only - decision-point telemetry
                    row["meta"] = meta
                card["rows"].append(row)
                completed.add(key)
                n_new += 1
                save()

                status = "OK " if ok else "XX "
                tag = ""
                if meta:
                    if meta.get("gate_hit"):
                        tag = " [GATED]"          # debate skipped - single-model answer
                    elif meta.get("cache_hit"):
                        tag = " [CACHE]"
                    elif meta.get("aborted"):
                        tag = " [ABORT]"
                    else:
                        bits = [f"r{meta.get('rounds', 0)}"]
                        if meta.get("drift_fired"):      bits.append("drift")
                        if meta.get("forced_challenge"): bits.append("chal")
                        if meta.get("free_mad"):         bits.append("freemad")
                        tag = " [" + ",".join(bits) + "]"
                print(f"[{bench:10} {arm:13} {j+1:2}/{len(tasks)}] "
                      f"{status} gold={task['gold']} pred={pred} "
                      f"conf={conf:.2f} {tok}tok ({dt:.0f}s){tag}"
                      + (f"  ERR {err}" if err else ""))

            # Recompute metrics from ALL rows for this arm (handles skipped rows correctly)
            _recompute_arm_metrics(card, bench, arm)
            save()

            acc  = card["accuracy"].get(bench, {}).get(arm, 0.0)
            cal  = card["calibration"].get(bench, {}).get(arm, {})
            cost = card["cost"].get(bench, {}).get(arm, {})
            suffix = f" ({n_new} new)" if n_new else " (all resumed)"
            print(f"--> {bench}/{arm}: acc={acc:.3f}  "
                  f"ece={cal.get('ece', 0):.3f}  "
                  f"{cost.get('mean_latency_s', 0):.0f}s/q  "
                  f"{cost.get('mean_tokens', 0):.0f}tok/q{suffix}\n")

        # Win-rates recomputed from rows after all arms for this bench are done
        _recompute_winrates(card, bench, arms)
        _recompute_gate_analysis(card, bench)
        save()

    # ── Routing: pure regex, always recompute from task set (free) ─────────────
    routing_pairs = [
        (t["mode_gold"], _answer_mode(t["route_query"]))
        for bench, tasks in tasks_by_bench.items()
        for t in tasks
    ]
    card["routing"] = M.routing_report(routing_pairs)
    save()

    # ── Robustness: skip if already populated ──────────────────────────────────
    robustness_done = bool(card.get("robustness"))
    if not args.no_robustness and "swarm" in arms and tasks_by_bench and not robustness_done:
        card["robustness"] = run_robustness(tasks_by_bench, card)
        save()
    elif robustness_done:
        print("[resume] Robustness already computed - skipping.")

    _print_card(card)
    save()
    print(f"\nSaved: {out_path}")


def _print_card(card):
    print("\n" + "=" * 64 + "\n EVAL CARD\n" + "=" * 64)
    for bench in card["accuracy"]:
        print(f"\n[{bench}]")
        for arm in card["accuracy"][bench]:
            acc  = card["accuracy"][bench][arm]
            cal  = card["calibration"][bench][arm]
            cost = card["cost"][bench][arm]
            print(f"  {arm:16} acc={acc:.3f}  ece={cal['ece']:.3f} brier={cal['brier']:.3f}  "
                  f"{cost['mean_latency_s']:.0f}s/q  {cost['mean_tokens']:.0f}tok/q")
        wr = card["winrate"].get(bench)
        if wr:
            print(f"  win-rate: swarm rescued {wr['only_swarm']}, "
                  f"regressed {wr['only_baseline']} "
                  f"(Δacc={wr['swarm_minus_baseline']:+.3f}, McNemar p={wr['mcnemar_p']})")
        wr_sc = card["winrate"].get(f"{bench}_vs_sc")
        if wr_sc:
            print(f"  win-rate swarm vs SC: swarm rescued {wr_sc['only_swarm']}, "
                  f"SC rescued {wr_sc['only_baseline']} "
                  f"(Δacc swarm-SC={wr_sc['swarm_minus_baseline']:+.3f}, "
                  f"McNemar p={wr_sc['mcnemar_p']})")
        # Gate analysis: how much of the swarm arm was actually the debate?
        ga = card.get("gate_analysis", {}).get(bench)
        if ga:
            ad = f"{ga['acc_debated']:.3f}" if ga['acc_debated'] is not None else "n/a"
            ag_ = f"{ga['acc_gated']:.3f}" if ga['acc_gated'] is not None else "n/a"
            print(f"  gate: fired {ga['n_gated']}/{ga['n_total']} "
                  f"({ga['gate_fire_rate']:.0%}) - "
                  f"acc debated-only={ad}  acc gated-only={ag_}  "
                  f"overall={ga['acc_overall']:.3f}")
        wr_ge = card["winrate"].get(f"{bench}_gate_effect")
        if wr_ge:
            print(f"  gate effect (vs no-gate): gate HARMED {wr_ge['only_swarm']}, "
                  f"gate HELPED {wr_ge['only_baseline']} "
                  f"(Δacc nogate-gate={wr_ge['swarm_minus_baseline']:+.3f}, "
                  f"McNemar p={wr_ge['mcnemar_p']})")
        rb = card["robustness"].get(bench)
        if rb:
            print(f"  robustness (Jetson down): {rb['full_acc']:.2f} -> "
                  f"{rb['degraded_acc']:.2f} (retention {rb['retention']:.2f})")
    rt = card.get("routing", {})
    if rt:
        print(f"\n[routing]  _answer_mode accuracy={rt.get('accuracy', 0):.3f} "
              f"(n={rt.get('n', 0)})")
        for cls, s in rt.get("per_class", {}).items():
            print(f"    {cls:8} P={s['precision']:.2f} R={s['recall']:.2f} "
                  f"F1={s['f1']:.2f} (support {s['support']})")


if __name__ == "__main__":
    main()
