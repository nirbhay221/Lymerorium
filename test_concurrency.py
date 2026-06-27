"""
Concurrency test - fires 2 Tier 2 queries simultaneously at /swarm_chat
and measures if the server handles them without blocking or crashing.
"""

import requests
import time
import threading

BASE = "http://localhost:5000"
TIMEOUT = 400

QUERIES = [
    "What are the risks of deploying LLMs on low-power edge devices?",
    "How does quantization affect the accuracy of vision models on embedded hardware?",
]

results = {}


def fire(idx: int, query: str) -> None:
    t0 = time.time()
    try:
        resp = requests.post(
            f"{BASE}/swarm_chat",
            json={"query": query},
            timeout=TIMEOUT,
        )
        elapsed = round(time.time() - t0, 1)
        data = resp.json()
        results[idx] = {
            "ok": True,
            "tier": data.get("tier"),
            "agents": data.get("agents_used", []),
            "elapsed": elapsed,
            "server_time": data.get("response_time_s"),
            "answer_preview": (data.get("answer") or "")[:200],
        }
    except Exception as e:
        results[idx] = {"ok": False, "error": str(e), "elapsed": round(time.time() - t0, 1)}


if __name__ == "__main__":
    print("\nConcurrency test - 2 simultaneous Tier 2 requests")
    print("="*60)
    for i, q in enumerate(QUERIES):
        print(f"  [{i}] {q[:70]}")
    print("="*60)

    threads = [threading.Thread(target=fire, args=(i, q)) for i, q in enumerate(QUERIES)]

    wall_start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=TIMEOUT + 10)
    wall_elapsed = round(time.time() - wall_start, 1)

    print(f"\nBoth requests completed in {wall_elapsed}s wall time\n")

    for i, q in enumerate(QUERIES):
        r = results.get(i, {"ok": False, "error": "no result"})
        print(f"  Request [{i}]: {'OK' if r['ok'] else 'FAILED'}")
        if r["ok"]:
            print(f"    Tier      : {r['tier']}")
            print(f"    Agents    : {r['agents']}")
            print(f"    Time      : {r['elapsed']}s (server: {r['server_time']}s)")
            print(f"    Answer    : {r['answer_preview'][:150].replace(chr(10), ' ')}")
        else:
            print(f"    Error     : {r.get('error')}")
        print()

    # Verdict
    both_ok = all(results.get(i, {}).get("ok") for i in range(len(QUERIES)))
    if both_ok:
        times = [results[i]["elapsed"] for i in range(len(QUERIES))]
        sequential_estimate = sum(times)
        print(f"  VERDICT: Both requests succeeded concurrently.")
        print(f"  Wall time {wall_elapsed}s vs sequential estimate ~{sequential_estimate}s")
        if wall_elapsed < sequential_estimate * 0.75:
            print("  Genuine parallelism - requests ran simultaneously.")
        else:
            print("  Requests likely serialized by the single-GPU inference bottleneck (expected).")
    else:
        print("  VERDICT: One or more requests failed.")
