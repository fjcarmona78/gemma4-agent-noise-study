"""Classify why each failed task run failed, from the agent's patch, the hidden-test output,
the reference patch and the problem statement.

Usage: failure_triage.py RUN [RUN ...]   (names under results/)
"""
import collections
import json
import re
import sys
from pathlib import Path

P = Path.home() / 'projects' / 'gemma-agent'
tasks = {}
for _f in ('data/tasks.jsonl', 'data/rebench_scale/tasks.jsonl', 'data/rebench_v2/tasks.jsonl', 'data/rebench_recent/tasks.jsonl'):
    if (P / _f).exists():  # dev91 plus the rebench sets; the first file listed wins for a duplicated id
        for _l in (P / _f).read_text().splitlines():
            _t = json.loads(_l)
            tasks.setdefault(_t['instance_id'], _t)


def changed_ranges(patch: str) -> dict[str, list[tuple[int, int]]]:
    """file -> list of (start, end) line ranges touched in the ORIGINAL file."""
    out = collections.defaultdict(list)
    current = None
    for line in patch.splitlines():
        if line.startswith('--- a/'):
            current = line[6:].strip()
        elif line.startswith('--- /dev/null'):
            current = None
        elif line.startswith('+++ b/') and current is None:
            current = line[6:].strip()
        elif line.startswith('@@') and current:
            m = re.match(r'@@ -(\d+)(?:,(\d+))? ', line)
            if m:
                start, n = int(m.group(1)), int(m.group(2) or 1)
                out[current].append((start, start + max(n, 1)))
    return out


def is_test_file(path: str) -> bool:
    parts = path.split('/')
    return any(p in ('tests', 'test') for p in parts[:-1]) or parts[-1].startswith('test_')


def classify(run: str, row: dict) -> dict:
    task = tasks[row['instance_id']]
    patch_file = P / 'results' / run / 'patches' / f"{row['instance_id']}.patch"
    agent_patch = patch_file.read_text(errors='ignore') if patch_file.exists() else ''
    out_file = P / 'results' / run / 'test_outputs' / f"{row['instance_id']}.log"
    test_out = out_file.read_text(errors='ignore') if out_file.exists() else ''
    gold = {f: r for f, r in changed_ranges(task['patch']).items() if not is_test_file(f)}
    mine = {f: r for f, r in changed_ranges(agent_patch).items() if not is_test_file(f)}
    right_file = bool(set(gold) & set(mine))
    near = False
    for f in set(gold) & set(mine):
        for a0, a1 in mine[f]:
            for g0, g1 in gold[f]:
                if a0 <= g1 + 15 and g0 <= a1 + 15:  # same region (within 15 lines)
                    near = True
    timed_out = 'timeout' in (row.get('error') or '').lower()
    # first exception type / missing name from the hidden-test output
    exc = None
    for name in ('ImportError', 'ModuleNotFoundError', 'AttributeError', 'NameError', 'TypeError',
                 'AssertionError', 'ValueError', 'KeyError', 'SyntaxError', 'IndentationError'):
        if re.search(rf'\bE\s+{name}\b|\b{name}:', test_out):
            exc = name
            break
    missing = None
    m = (re.search(r"cannot import name '([^']+)'", test_out) or re.search(r"has no attribute '([^']+)'", test_out)
         or re.search(r"name '([^']+)' is not defined", test_out) or re.search(r"unexpected keyword argument '([^']+)'", test_out))
    if m:
        missing = m.group(1)
    statement = task['problem_statement']
    spec_gap = bool(missing and missing not in statement)
    if not mine:
        category = 'no source change' + (' (timed out)' if timed_out else ' (gave up/submitted empty)')
    elif not right_file:
        category = 'wrong file'
    elif missing and spec_gap:
        category = 'needs a name not in the statement'
    elif not near:
        category = 'right file, different place'
    elif timed_out:
        category = 'right place, ran out of time'
    else:
        category = 'right place, wrong fix (submitted)'
    statement_has_code = bool(re.search(r'`[^`]+`|\w+\(|\w+_\w+|[a-z][A-Z]\w+', statement))
    return {'run': run, 'task': row['instance_id'], 'category': category, 'timed_out': timed_out, 'exc': exc,
            'missing': missing, 'spec_gap': spec_gap, 'right_file': right_file, 'near': near,
            'gold_lines': sum(e - s for r in gold.values() for s, e in r), 'statement_has_code': statement_has_code}


rows = []
for run in sys.argv[1:]:
    for line in (P / 'results' / run / 'task_results.jsonl').read_text().splitlines():
        r = json.loads(line)
        if not r['resolved']:
            rows.append(classify(run, r))
print(f'failed task runs: {len(rows)} (from {len(sys.argv) - 1} runs)')
cats = collections.Counter(r['category'] for r in rows)
for c, n in cats.most_common():
    ex = collections.Counter(r['exc'] for r in rows if r['category'] == c).most_common(3)
    print(f'  {n:4d} ({100 * n / len(rows):4.1f}%)  {c:40s} test errors: {ex}')
print('\nper task (failed runs / category of the most common failure):')
per_task = collections.defaultdict(list)
for r in rows:
    per_task[r['task']].append(r['category'])
json.dump(rows, open(Path(sys.argv[0]).with_name('failure_rows.json'), 'w'), indent=1)
