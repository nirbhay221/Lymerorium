"""Full distributed swarm: drives api.run_simulation and tallies per-NODE participation."""
import sys, os, time, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "swarm_core"))
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import config
from api import run_simulation
from simulation import _AGENT_LANE

LANE_NODE = {
    "reasoning": f"Qwen  laptop-GPU  ({config.get_reasoning_llm_config()['model']})",
    "diversity": f"Llama laptop-CPU  ({config.get_diversity_llm_config()['model']})",
    "vision":    f"Gemma Jetson:8080 ({config.get_vision_llm_config()['model']})",
}

# Topic must be semantically distant from cached debates (cache uses cosine similarity, threshold
# 0.42 - a timestamp suffix alone doesn't help because the embedding captures the full text).
# Using a rotating set of off-domain topics keyed on the minute so consecutive runs differ.
_TOPICS = [
    "Should cities ban private car ownership in their historic centres?",
    "Is nuclear fission the most viable path to carbon-neutral baseload power?",
    "Does universal basic income reduce or increase the incentive to work?",
    "Should competitive sports be stratified by biological sex or by hormone levels?",
    "Is space colonisation a moral obligation or an irresponsible use of resources?",
]
topic = _TOPICS[int(time.time()) // 60 % len(_TOPICS)]
print("=" * 70)
print("FULL DISTRIBUTED SWARM -", topic)
print("=" * 70)

t0 = time.time()
r = run_simulation(topic, max_rounds=2)
dt = time.time() - t0

# Tally which agents spoke, mapped to their pinned node.
speakers = collections.Counter()
for m in r.get("messages", []):
    name = m.get("agent") or m.get("name") or m.get("role")
    if name in _AGENT_LANE:
        speakers[name] += 1

node_calls = collections.Counter()
for name, n in speakers.items():
    node_calls[_AGENT_LANE[name]] += n

print(f"\nwall_time        : {dt:.1f}s")
print(f"rounds_completed : {r.get('rounds_completed')}")
print(f"convergence      : {r.get('convergence_score')}")
print(f"cache_hit        : {r.get('cache_hit')}")
print("\nPER-NODE PARTICIPATION (proves the debate spanned all 3 machines):")
for lane in ("reasoning", "diversity", "vision"):
    calls = node_calls.get(lane, 0)
    who = ", ".join(f"{n}x{c}" for n, c in speakers.items() if _AGENT_LANE[n] == lane) or "-"
    flag = "OK" if calls else "NONE"
    print(f"  [{flag:4}] {LANE_NODE[lane]:42} {calls:2} msgs  ({who})")

print("\n" + "-" * 70)
print("VERDICT:")
print(r.get("verdict", "")[:1500])
print("=" * 70)
