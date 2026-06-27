"""One real 3-node debate: Jetson Gemma + laptop Qwen(GPU) + laptop Llama(CPU)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "swarm_core"))
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import config
from oracle_chat import answer

print("=" * 64)
print("3-NODE SWARM CONFIG")
print("  Vision/Gemma :", config.get_vision_llm_config()["base_url"], "|", config.get_vision_llm_config()["model"])
print("  Reasoning    :", config.get_reasoning_llm_config()["base_url"], "|", config.get_reasoning_llm_config()["model"])
print("  Diversity    :", config.get_diversity_llm_config()["base_url"], "|", config.get_diversity_llm_config()["model"])
print("=" * 64)

q = "Should governments regulate large language models?"
print(f"\nQUERY: {q}\n")
t0 = time.time()
r = answer(q, None, background_swarm=None)
dt = time.time() - t0

print("-" * 64)
print("  tier        :", r.get("tier"))
print("  agents_used :", r.get("agents_used"))
print("  wall_time   :", f"{dt:.1f}s")
print("-" * 64)
print(r.get("answer", r.get("error", ""))[:1200])
print("=" * 64)
