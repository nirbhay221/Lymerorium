import requests, json, time

BASE = "http://localhost:5000"

# Test Tier 3: complex question - should return Tier 2 immediately + background full swarm
print("=== T3: complex - non-blocking ===")
t0 = time.time()
r = requests.post(BASE + "/swarm_chat",
    json={"query": "Should humanity colonize Mars, and what are the existential risks?", "vision": False},
    timeout=180)
d = r.json()
print(f"Tier: {d.get('tier')} | Time: {d.get('response_time_s')}s")
print(f"Note: {d.get('note', 'none')}")
print(f"Answer: {d.get('answer', d.get('error', ''))[:400]}")

# Wait a bit and check if background swarm stored the verdict
print("\n[Waiting 5s for background swarm to register...]")
time.sleep(5)

r2 = requests.get(BASE + "/swarm_status", timeout=5)
print("Status after T3:", json.dumps(r2.json(), indent=2))
