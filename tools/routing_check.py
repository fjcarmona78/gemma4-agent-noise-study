"""Routing headroom check (CPU, existing runs only).
1) Split-half: does a task's preferred workflow on half its runs predict the other half?
2) Feature router: depth-<=2 tree on pre-solve statement features, leave-one-repo-out."""
import json, glob, re, random, itertools, collections
from pathlib import Path
P = Path(__file__).resolve().parents[1]
SETS = {'v2t74': 74, 'recent': 33, 'ood45': 45, 'fresh15': 15, 'rebench_dev': 53}
POL = {'bundle': 'public_bav2_noedit_t7', 'safe': 'v4c_pinned_cap28_noedit_safe'}
runs = collections.defaultdict(lambda: collections.defaultdict(list))  # task -> pol -> [0/1]
for s, n in SETS.items():
    for pol, stem in POL.items():
        for d in sorted(x for x in glob.glob(str(P / 'results' / f'{s}_{stem}')) + glob.glob(str(P / 'results' / f'{s}_{stem}_rep*')) if Path(x, 'task_results.jsonl').is_file()):
            rows = [json.loads(l) for l in open(f'{d}/task_results.jsonl')]
            if len(rows) != n: continue
            for r in rows: runs[r['instance_id']][pol].append(int(bool(r['resolved'])))
tasks = {}
for f in glob.glob(str(P / 'data/*/tasks.jsonl')):
    for l in open(f):
        t = json.loads(l); tasks.setdefault(t['instance_id'], t)
ids = [t for t in runs if len(runs[t]['bundle']) >= 2 and len(runs[t]['safe']) >= 2]
print('tasks with >=2 runs of each:', len(ids), 'of', len(runs))
rate = lambda xs: sum(xs) / len(xs)
B = sum(rate(runs[t]['bundle']) for t in ids); S = sum(rate(runs[t]['safe']) for t in ids)
O = sum(max(rate(runs[t]['bundle']), rate(runs[t]['safe'])) for t in ids)
print(f'expected solves: always-bundle {B:.1f}, always-safe {S:.1f}, in-sample oracle router {O:.1f} (optimistic, fits noise)')
# split-half
random.seed(0); gains = []
for _ in range(2000):
    tot_route = tot_b = tot_s = 0
    for t in ids:
        b, s = runs[t]['bundle'][:], runs[t]['safe'][:]
        random.shuffle(b); random.shuffle(s)
        hb, hs = len(b) // 2, len(s) // 2
        bA, bB, sA, sB = b[:hb], b[hb:], s[:hs], s[hs:]
        pick = 'b' if rate(bA) > rate(sA) else 's' if rate(sA) > rate(bA) else random.choice('bs')
        tot_route += rate(bB) if pick == 'b' else rate(sB)
        tot_b += rate(bB); tot_s += rate(sB)
    gains.append(tot_route - max(tot_b, tot_s))
gains.sort()
print(f'split-half router minus best fixed policy, held-out half: mean {sum(gains)/len(gains):+.2f} tasks, '
      f'5-95% [{gains[100]:+.2f}, {gains[1900]:+.2f}]')
# features
def feats(st):
    st = st or ''
    return {'traceback': 'Traceback (most recent call last)' in st,
            'code_block': '```' in st,
            'error_name': bool(re.search(r'\b\w+(Error|Exception|Warning)\b', st)),
            'expected_actual': bool(re.search(r'(?i)expected|should (be|return)|instead', st)),
            'backtick_api': len(re.findall(r'`[\w.]+\(?\)?`', st)) >= 2,
            'long': len(st) > 1500}
X = {t: feats(tasks.get(t, {}).get('problem_statement')) for t in ids}
print('feature coverage:', {k: sum(X[t][k] for t in ids) for k in X[ids[0]]})
repo = lambda t: tasks.get(t, {}).get('repo') or t.rsplit('-', 1)[0]
diff = {t: rate(runs[t]['bundle']) - rate(runs[t]['safe']) for t in ids}
def fit(train):
    best = ('none', None, None, 0)  # rule = route to bundle where condition, safe elsewhere (or reverse)
    for k in X[ids[0]]:
        for val in (True, False):
            g = sum(diff[t] for t in train if X[t][k] == val)
            for pol_true in ('bundle', 'safe'):
                score = g if pol_true == 'bundle' else -g
                if score > best[3]: best = (k, val, pol_true, score)
    return best
base_b = sum(diff[t] > 0 for t in ids)
route_gain = 0.0; picks = collections.Counter()
by_repo = collections.defaultdict(list)
for t in ids: by_repo[repo(t)].append(t)
for r, test in by_repo.items():
    train = [t for t in ids if repo(t) != r]
    k, val, pol_true, _ = fit(train)
    # global default = better fixed policy on train
    default = 'bundle' if sum(diff[t] for t in train) > 0 else 'safe'
    picks[(k, val, pol_true)] += 1
    for t in test:
        pol = (pol_true if X[t][k] == val else ('safe' if pol_true == 'bundle' else 'bundle')) if k != 'none' else default
        route_gain += rate(runs[t][pol])
print(f'leave-one-repo-out 1-split router: {route_gain:.1f} expected solves vs always-bundle {B:.1f}, always-safe {S:.1f}')
print('rules chosen (top):', picks.most_common(4))
