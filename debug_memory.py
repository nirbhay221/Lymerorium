import sys
sys.path.insert(0, 'swarm_core')
from memory import MEMORY

print('Verdicts stored:', MEMORY.count)
print('Last topic:', MEMORY.last_topic)

result = MEMORY.search('what do you know about AI and robotics with NVIDIA?')
print('Search result (0.55 threshold):', result['topic'][:60] if result else 'NO HIT')

if MEMORY.count > 0:
    q = MEMORY._encode('what do you know about AI and robotics with NVIDIA?')
    idx = MEMORY._get_index()
    r = idx.search(q, 5)
    print('\nTop 5 nearest verdicts:')
    for item in r:
        v = MEMORY._verdicts.get(int(item.key), {})
        topic = str(v.get('topic', ''))[:55]
        sim = round(1.0 - float(item.distance), 3)
        print(f'  sim={sim}  topic={topic}')
