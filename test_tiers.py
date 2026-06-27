"""
Quick sanity test - fires one query per tier at /swarm_chat and prints results.
Tier 3 is tested via direct bypass (calls _tier3_complex internally) so we
don't depend on the LLM routing it correctly.
"""

import os
import sys
import requests
import time

BASE = "http://localhost:5000"
TIMEOUT = 400  # seconds

TIERS = [
    {
        "label": "TIER 1 - simple",
        "query": "What is the capital of France?",
    },
    {
        "label": "TIER 2 - medium",
        "query": "What are the main trade-offs between edge computing and cloud computing for real-time video AI?",
    },
]


def run_api_test(item: dict) -> None:
    label = item["label"]
    query = item["query"]
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Query: {query[:80]}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        resp = requests.post(
            f"{BASE}/swarm_chat",
            json={"query": query},
            timeout=TIMEOUT,
        )
        elapsed = round(time.time() - t0, 1)
        data = resp.json()

        tier_got    = data.get("tier", "?")
        agents_used = data.get("agents_used", [])
        answer      = (data.get("answer") or "").strip()
        note        = data.get("note", "")

        print(f"  Tier routed to : {tier_got}")
        print(f"  Agents used    : {agents_used}")
        print(f"  Response time  : {elapsed}s (server says {data.get('response_time_s','?')}s)")
        if note:
            print(f"  Note           : {note}")
        print(f"\n  Answer (first 400 chars):\n")
        print("  " + answer[:400].replace("\n", "\n  "))

    except requests.exceptions.Timeout:
        elapsed = round(time.time() - t0, 1)
        print(f"  TIMEOUT after {elapsed}s")
    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        print(f"  ERROR after {elapsed}s: {e}")


def run_tier3_bypass() -> None:
    """Directly call _tier3_complex - bypasses the severity router entirely."""
    query = "Should autonomous AI surveillance systems be deployed in public spaces, and what governance frameworks would make them safe?"

    print(f"\n{'='*60}")
    print(f"  TIER 3 - complex (DIRECT BYPASS)")
    print(f"  Query: {query[:80]}")
    print(f"{'='*60}")

    # Wire up swarm_core path
    swarm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swarm_core")
    if swarm_path not in sys.path:
        sys.path.insert(0, swarm_path)

    try:
        from oracle_chat import _tier3_complex
        t0 = time.time()
        # background_swarm=None: swarm stores result in MEMORY but no pause/resume
        result = _tier3_complex(query, image_b64="", background_swarm=None)
        elapsed = round(time.time() - t0, 1)

        tier_got    = result.get("tier", "?")
        agents_used = result.get("agents_used", [])
        answer      = (result.get("answer") or "").strip()
        note        = result.get("note", "")

        print(f"  Tier returned  : {tier_got}")
        print(f"  Agents used    : {agents_used}")
        print(f"  Response time  : {elapsed}s (preliminary - full swarm running in bg)")
        if note:
            print(f"  Note           : {note}")
        print(f"\n  Preliminary answer (first 400 chars):\n")
        print("  " + answer[:400].replace("\n", "\n  "))

    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        print(f"  ERROR after {elapsed}s: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    print("\nTesting 3-tier oracle_chat")
    print("Timeout per API request:", TIMEOUT, "seconds")

    for item in TIERS:
        run_api_test(item)

    run_tier3_bypass()

    print("\n\nDone.")
