import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "swarm_core"))
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

from oracle_chat import answer

SEP = "-" * 60

print(SEP)
print("TEST 1: Tier 1 - simple factual")
r = answer("what is the speed of light?", None)
print("  tier   :", r["tier"])
print("  agents :", r["agents_used"])
print("  answer :", r["answer"][:200])

print(SEP)
print("TEST 2: Tier 2 - medium, multi-agent")
r = answer("what are the main challenges of deploying AI on edge devices?", None)
print("  tier   :", r["tier"])
print("  agents :", r["agents_used"])
print("  answer :", r["answer"][:200])

print(SEP)
print("TEST 3: iMAD gate - complex, check if debate fires or skips")
r = answer("should governments regulate large language models?", None, background_swarm=None)
print("  tier   :", r["tier"])
print("  agents :", r["agents_used"])
print("  answer :", r["answer"][:200])

print(SEP)
print("TEST 4: Memory hit - query matching stored verdict")
from memory import MEMORY
past = MEMORY.search("robotics AI edge computing")
print("  memory hit:", past["topic"][:80] if past else "None")
r = answer("tell me about robotics and AI edge computing", past)
print("  tier   :", r["tier"])
print("  answer :", r["answer"][:200])

print(SEP)
print("ALL TESTS DONE")
