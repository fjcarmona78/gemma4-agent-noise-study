"""CPU check of the second external review (2026-10-24): would an automatic existing-test gate or test-file reading help?
(1) Failure anatomy: for each failed attempt, are the failing hidden tests NEW/CHANGED by the task's test patch (spec the
    agent could not see) or UNTOUCHED pre-existing tests (a regression the repo's existing suite would have exposed)?
(2) Within-task behaviour: on tasks with both solved and failed runs, do solved runs read test files / run pytest more?
"""
import collections, glob, json, re
from pathlib import Path
P = Path(__file__).resolve().parents[1]
SETS = {'v2t74': 74, 'recent': 33, 'ood45': 45, 'fresh15': 15, 'rebench_dev': 53}
POL = {'bundle': 'public_bav2_noedit_t7', 'safe': 'v4c_pinned_cap28_noedit_safe'}
tasks = {}
for f in glob.glob(str(P / 'data/*/tasks.jsonl')):
    for l in open(f):
        t = json.loads(l); tasks.setdefault(t['instance_id'], t)
TESTFILE = re.compile(r'(^|/)(tests?/|test_[^/]*\.py$|[^/]*_test\.py$|conftest\.py)')
def flags(trace):
    f = dict(read_test=False, grep_tests=False, pytest=False)
    if not trace.exists(): return None
    for s in json.loads(trace.read_text())['steps']:
        if s.get('source') != 'agent': continue
        calls = s.get('tool_calls') or []
        if isinstance(calls, str): continue
        for c in calls:
            a = c.get('arguments') or {}
            if not isinstance(a, dict): continue
            fp, cmd = str(a.get('filepath') or a.get('path') or ''), str(a.get('command') or '')
            if c.get('function_name') == 'read_file' and TESTFILE.search(fp.lstrip('./')): f['read_test'] = True
            if c.get('function_name') == 'run_command':
                if re.search(r'\bpytest\b|unittest|tox\b', cmd): f['pytest'] = True
                if re.search(r'\b(grep|cat|sed -n|head|tail|rg)\b', cmd) and re.search(r'(^|\s|/)tests?/|test_\w+\.py', cmd): f['grep_tests'] = True
    return f
anat = collections.Counter(); only_ex = collections.Counter(); per_task = collections.defaultdict(list)
for s, n in SETS.items():
    for pol, stem in POL.items():
        for d in sorted(x for x in glob.glob(str(P / 'results' / f'{s}_{stem}')) + glob.glob(str(P / 'results' / f'{s}_{stem}_rep*')) if Path(x, 'task_results.jsonl').is_file()):
            rows = [json.loads(l) for l in open(f'{d}/task_results.jsonl')]
            if len(rows) != n: continue
            for r in rows:
                tid = r['instance_id']; fl = flags(Path(d, 'traces', f'trace_{tid}.json'))
                if fl and r.get("agent_patch_size"): per_task[tid].append((bool(r["resolved"]), fl))
                if r['resolved']: continue
                if not r.get('agent_patch_size'): anat['no patch'] += 1; continue
                log = Path(d, 'test_outputs', f'{tid}.log')
                if not log.exists(): anat['no log'] += 1; continue
                failed = set(re.findall(r'^(?:FAILED|ERROR) (\S+)', log.read_text(errors='ignore'), re.M))
                tp = tasks.get(tid, {}).get('test_patch') or ''
                touched_names = set(re.findall(r'def (test\w*)', tp))
                if not failed: anat['failed, no test ids in log (collection/import error etc.)'] += 1; continue
                names = {re.split(r'[\[]', x.split('::')[-1])[0] for x in failed}
                untouched = {x for x in names if x not in touched_names and x not in tp}
                if not untouched: anat['only new/changed hidden tests fail'] += 1
                elif untouched == names: anat['ONLY existing untouched tests break (gate fix would likely solve)'] += 1; only_ex[tid] += 1
                else: anat['existing tests break AND new hidden tests fail'] += 1
tot = sum(anat.values())
print('FAILED ATTEMPTS, bundle + safe, all complete runs:', tot)
for k, v in anat.most_common(): print(f'  {v:4d} {v/tot:5.0%}  {k}')
print('  regression-only failures spread over', len(only_ex), 'tasks; top:', only_ex.most_common(6))
mixed = {t: v for t, v in per_task.items() if any(x[0] for x in v) and not all(x[0] for x in v)}
print(f'\nTASKS WITH BOTH SOLVED AND FAILED RUNS: {len(mixed)} (within-task averages)')
for k in ('read_test', 'grep_tests', 'pytest'):
    ws = [sum(f[k] for ok, f in v if ok) / sum(ok for ok, _ in v) for v in mixed.values()]
    ls = [sum(f[k] for ok, f in v if not ok) / sum(not ok for ok, _ in v) for v in mixed.values()]
    more_w = sum(a > b for a, b in zip(ws, ls)); more_l = sum(b > a for a, b in zip(ws, ls))
    print(f'  {k:10s} solved runs {sum(ws)/len(ws):5.0%}  failed runs {sum(ls)/len(ls):5.0%}  tasks where solved runs do it more {more_w}, less {more_l}')
