import sys
sys.path.insert(0, 'swarm_core')
from oracle_chat import _COMPLEX_RE, _SIMPLE_RE

tests = [
    ('what is machine learning?', 'simple'),
    ('what time is it?', 'simple'),
    ('should governments regulate AI development?', 'complex'),
    ('what is the future of humanity if AGI is developed?', 'complex'),
    ('will AI cause human extinction?', 'complex'),
    ('is Python better than JavaScript?', 'medium'),
    ('what are the best practices for REST APIs?', 'medium'),
    ('should we ban AI from warfare?', 'complex'),
    ('ethics of using AI in hiring decisions', 'complex'),
    ('will AGI destroy society?', 'complex'),
    ('what is the best database for a startup?', 'medium'),
    ('explain neural networks', 'simple'),
]

print('REGEX ROUTING TEST')
print('-' * 60)
all_pass = True
for q, expected in tests:
    words = q.split()
    if _COMPLEX_RE.search(q):
        got = 'complex'
    elif len(words) <= 12 and _SIMPLE_RE.match(q):
        got = 'simple'
    else:
        got = 'medium(LLM)'
    ok = 'PASS' if got.startswith(expected) else 'FAIL'
    if not got.startswith(expected):
        all_pass = False
    print(f'  {ok} [{expected:7}] -> [{got:12}]  {q[:55]}')

print()
print('ALL PASS' if all_pass else 'SOME FAILED - check above')
