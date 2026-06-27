"""
Industry-standard benchmark: single-model baseline vs. full 3-node swarm.
Benchmarks: GSM8K (math reasoning, numeric gold) + TruthfulQA MC1 (factuality, 1 correct choice).

Methodology: same sampled questions through both arms, accuracy + latency reported, n disclosed.

Fairness: BOTH arms get the SAME uniform LLM extractor on their raw output, so neither
arm is advantaged by answer-format prompting. Baseline = one qwen3:8b pass. Swarm =
api.run_simulation (Gemma/Jetson + Qwen/GPU + Llama/CPU debate).

Usage: python bench/run_bench.py --bench gsm8k,truthfulqa --n 10 --arms baseline,swarm
       python bench/run_bench.py --smoke      # n=1, baseline only, fast plumbing check
"""
import sys, os, re, json, time, random, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "swarm_core"))
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import requests
import config  # noqa: F401  (forces UTF-8 stdout + loads .env)
from api import run_simulation

HERE = os.path.dirname(os.path.abspath(__file__))
QWEN_URL = "http://localhost:11434/v1/chat/completions"
QWEN_MODEL = "qwen3:8b"        # baseline reasoning arm
EXTRACT_MODEL = "llama3.1:8b"  # independent NON-thinking extractor (fair to both arms)
SEED = 42


def strip_think(t: str) -> str:
    return re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL | re.IGNORECASE).strip()


def chat(prompt: str, system: str = "", max_tokens: int = 1024, temperature: float = 0.2,
         model: str = QWEN_MODEL) -> str:
    """One single-model call (baseline arm uses Qwen; extractor uses non-thinking Llama)."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens,
            "temperature": temperature, "stream": False}
    r = requests.post(QWEN_URL, json=body, timeout=300)
    r.raise_for_status()
    return strip_think(r.json()["choices"][0]["message"]["content"])


# Uniform extractors (same for both arms)

def extract_number(text: str):
    """LLM-extract the final numeric answer from any solution/verdict text."""
    out = chat(
        "Below is a worked solution to a math word problem. Output ONLY the final "
        "numeric answer (digits only, no words, no units, no currency symbol).\n\n"
        f"Solution:\n{text[-2500:]}\n\nFinal numeric answer:",
        max_tokens=50, temperature=0.0, model=EXTRACT_MODEL,
    )
    nums = re.findall(r"-?\d[\d,]*\.?\d*", out)
    if not nums:
        return None
    return _norm_num(nums[-1])


def _norm_num(s: str):
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        f = float(s)
        return int(f) if f.is_integer() else round(f, 4)
    except ValueError:
        return None


def gsm_gold(answer: str):
    return _norm_num(answer.split("####")[-1].strip())


def extract_letter(text: str, options_block: str):
    """LLM-extract which option letter the text concludes is correct."""
    out = chat(
        "A question had these lettered options:\n"
        f"{options_block}\n\n"
        "Below is an analysis/answer. Output ONLY the single letter (A, B, C, ...) of the "
        "option it concludes is the best, most truthful answer.\n\n"
        f"Analysis:\n{text[-2500:]}\n\nLetter:",
        max_tokens=10, temperature=0.0, model=EXTRACT_MODEL,
    )
    m = re.search(r"[A-Z]", out.upper())
    return m.group(0) if m else None


# Benchmark loaders -> list of normalized task dicts

def load_gsm8k(n):
    data = json.load(open(os.path.join(HERE, "gsm8k_test.json"), encoding="utf-8"))
    idx = random.Random(SEED).sample(range(len(data)), n)
    out = []
    for i in idx:
        d = data[i]
        out.append({"id": i, "question": d["question"], "gold": gsm_gold(d["answer"])})
    return out


def load_truthfulqa(n):
    data = json.load(open(os.path.join(HERE, "truthfulqa_mc.json"), encoding="utf-8"))
    idx = random.Random(SEED).sample(range(len(data)), n)
    out = []
    for i in idx:
        d = data[i]
        targets = d["mc1_targets"]  # {choice_text: 0/1}, exactly one 1
        choices = list(targets.items())
        random.Random(SEED + i).shuffle(choices)
        letters = [chr(65 + k) for k in range(len(choices))]
        block = "\n".join(f"{letters[k]}. {c}" for k, (c, _) in enumerate(choices))
        gold_letter = next(letters[k] for k, (_, lab) in enumerate(choices) if lab == 1)
        out.append({"id": i, "question": d["question"], "options_block": block,
                    "gold": gold_letter})
    return out


# Arms

def run_gsm8k_item(task, arm):
    q = task["question"]
    if arm == "baseline":
        raw = chat(f"Solve this math word problem step by step. End with the final number.\n\n"
                   f"Problem: {q}", max_tokens=2048, temperature=0.0)
    else:
        # answer_mode="numeric": swarm agents keep chain-of-thought, emit FINAL: lines,
        # the verdict is a majority vote, and the consensus-hostile Contrarian/FREE-MAD
        # machinery is suppressed.
        res = run_simulation(q, max_rounds=2, answer_mode="numeric")
        raw = res.get("verdict", "")
    pred = extract_number(raw)
    return pred, raw


def run_truthfulqa_item(task, arm):
    q, block = task["question"], task["options_block"]
    prompt = (f"Question: {q}\n\nOptions:\n{block}\n\n"
              "Select the single most truthful and factually accurate option.")
    if arm == "baseline":
        raw = chat(prompt + " Reply with the letter and a one-sentence reason.",
                   max_tokens=512, temperature=0.0)
    else:
        # answer_mode="choice": agents emit FINAL: <letter>, verdict is a majority vote.
        res = run_simulation(prompt, max_rounds=2, answer_mode="choice")
        raw = res.get("verdict", "")
    pred = extract_letter(raw, block)
    return pred, raw


RUNNERS = {"gsm8k": run_gsm8k_item, "truthfulqa": run_truthfulqa_item}
LOADERS = {"gsm8k": load_gsm8k, "truthfulqa": load_truthfulqa}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="gsm8k,truthfulqa")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--arms", default="baseline,swarm")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n, args.arms = 1, "baseline"

    benches = args.bench.split(",")
    arms = args.arms.split(",")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(HERE, f"results_{stamp}.json")
    results = {"meta": {"n": args.n, "arms": arms, "benches": benches, "seed": SEED,
                        "started": stamp}, "rows": []}

    def save():
        json.dump(results, open(out_path, "w", encoding="utf-8"), indent=2)

    print(f"=== BENCHMARK RUN  n={args.n}  arms={arms}  benches={benches} ===")
    print(f"Results -> {out_path}\n")

    for bench in benches:
        tasks = LOADERS[bench](args.n)
        runner = RUNNERS[bench]
        for arm in arms:
            correct = 0
            lat = []
            for j, task in enumerate(tasks):
                t0 = time.time()
                try:
                    pred, raw = runner(task, arm)
                    err = None
                except Exception as e:
                    pred, raw, err = None, "", f"{type(e).__name__}: {e}"
                dt = time.time() - t0
                lat.append(dt)
                ok = (pred is not None and str(pred) == str(task["gold"]))
                correct += int(ok)
                row = {"bench": bench, "arm": arm, "id": task["id"], "gold": task["gold"],
                       "pred": pred, "correct": ok, "latency_s": round(dt, 1), "error": err}
                results["rows"].append(row)
                save()
                flag = "OK " if ok else "XX "
                print(f"[{bench:10} {arm:8} {j+1:2}/{len(tasks)}] {flag} "
                      f"gold={task['gold']} pred={pred} ({dt:.0f}s)"
                      + (f"  ERR {err}" if err else ""))
            acc = correct / len(tasks) * 100
            mean_lat = sum(lat) / len(lat)
            summ = {"bench": bench, "arm": arm, "accuracy_pct": round(acc, 1),
                    "correct": correct, "n": len(tasks), "mean_latency_s": round(mean_lat, 1)}
            results.setdefault("summary", []).append(summ)
            save()
            print(f"--> {bench}/{arm}: {acc:.1f}% ({correct}/{len(tasks)})  "
                  f"mean {mean_lat:.0f}s/q\n")

    print("=== SUMMARY ===")
    for s in results.get("summary", []):
        print(f"  {s['bench']:10} {s['arm']:8} {s['accuracy_pct']:5.1f}%  "
              f"({s['correct']}/{s['n']})  {s['mean_latency_s']:.0f}s/q")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
