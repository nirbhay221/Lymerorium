"""Fetch GSM8K (test) and TruthfulQA (multiple-choice) from official repos -> local JSON."""
import json, os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(HERE, exist_ok=True)

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
# Official TruthfulQA multiple-choice task (MC1/MC2 targets)
TQA_URLS = [
    "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/data/mc_task.json",
    "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.json",
]

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")

# GSM8K
print("Fetching GSM8K test...")
raw = get(GSM8K_URL)
gsm = [json.loads(l) for l in raw.splitlines() if l.strip()]
json.dump(gsm, open(os.path.join(HERE, "gsm8k_test.json"), "w", encoding="utf-8"))
print(f"  GSM8K: {len(gsm)} items | sample answer tail: ...{gsm[0]['answer'][-30:]!r}")

# TruthfulQA MC
tqa = None
for u in TQA_URLS:
    try:
        print(f"Fetching TruthfulQA MC from {u} ...")
        tqa = json.loads(get(u))
        break
    except Exception as e:
        print(f"  failed: {e}")
if tqa is None:
    print("TruthfulQA MC fetch FAILED on all URLs"); sys.exit(1)
json.dump(tqa, open(os.path.join(HERE, "truthfulqa_mc.json"), "w", encoding="utf-8"))
ex = tqa[0]
print(f"  TruthfulQA: {len(tqa)} items | fields: {list(ex.keys())}")
mc1 = ex.get("mc1_targets", {})
print(f"  mc1 choices: {len(mc1.get('choices', []))} | labels: {mc1.get('labels', [])[:8]}")


# StrategyQA + MMLU via the HuggingFace datasets-server (public JSON rows API, no auth).

def hf_rows(dataset, config_, split, length=400, offset=0):
    """Pull up to `length` rows from the datasets-server (max 100/page, so we page)."""
    out = []
    while len(out) < length:
        page = min(100, length - len(out))
        url = (f"https://datasets-server.huggingface.co/rows?dataset={dataset}"
               f"&config={config_}&split={split}&offset={offset+len(out)}&length={page}")
        rows = json.loads(get(url)).get("rows", [])
        if not rows:
            break
        out.extend(r["row"] for r in rows)
    return out


# StrategyQA - yes/no multi-hop commonsense. Normalize to {question, answer: "yes"/"no"}.
print("Fetching StrategyQA...")
sqa = None
for ds, cfg, split in [("ChilleD/StrategyQA", "default", "test"),
                       ("ChilleD/StrategyQA", "default", "train"),
                       ("tasksource/strategy-qa", "default", "train")]:
    try:
        rows = hf_rows(ds, cfg, split, length=400)
        if rows:
            sqa = [{"question": r.get("question") or r.get("inputs"),
                    "answer": "yes" if (r.get("answer") in (True, "True", "true", 1, "yes")) else "no"}
                   for r in rows if (r.get("question") or r.get("inputs"))]
            print(f"  StrategyQA from {ds}/{split}: {len(sqa)} items")
            break
    except Exception as e:
        print(f"  {ds}/{split} failed: {e}")
if sqa:
    json.dump(sqa, open(os.path.join(HERE, "strategyqa.json"), "w", encoding="utf-8"))
else:
    print("  StrategyQA fetch FAILED - eval_card will skip this bench")

# MMLU - broad-knowledge 4-way MC. Normalize to {question, choices:[...], answer:int}.
print("Fetching MMLU (test split, 'all' config)...")
try:
    rows = hf_rows("cais/mmlu", "all", "test", length=400)
    mmlu = [{"question": r["question"], "choices": r["choices"], "answer": int(r["answer"])}
            for r in rows if r.get("choices") and len(r["choices"]) >= 2]
    json.dump(mmlu, open(os.path.join(HERE, "mmlu.json"), "w", encoding="utf-8"))
    print(f"  MMLU: {len(mmlu)} items")
except Exception as e:
    print(f"  MMLU fetch FAILED ({e}) - eval_card will skip this bench")

print("DONE")
