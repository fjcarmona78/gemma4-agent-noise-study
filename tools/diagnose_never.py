"""Structured diagnosis of the dev45 tasks that v3 never solves.

For each task: gold-fix size and files, names the hidden tests import from the package
(and whether the repo or problem statement already contains them), and per-run outcomes
across all dev45 experiments: time used, edits, files touched vs gold, failure messages.
"""

import io
import json
import re
import sys
import tarfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / 'results'
RUNS = ['step1C_v3_dev91_t4', 'dev45_v3_rerun', 'dev45_think2048', 'dev45_think512',
        'dev45_timeaware', 'dev45_nothink']
PKG = {'fastapi/fastapi': 'fastapi', 'Textualize/rich': 'rich', 'psf/requests': 'requests',
       'encode/httpx': 'httpx'}

tasks = {json.loads(l)['instance_id']: json.loads(l) for l in open(PROJECT / 'data/tasks.jsonl')}
never = json.load(open(RESULTS / 'dev45_categories.json'))['never']


def diff_files(patch: str) -> list[str]:
    return re.findall(r'^\+\+\+ b/(\S+)', patch, re.M)


def diff_size(patch: str) -> tuple[int, int]:
    added = sum(1 for l in patch.splitlines() if l.startswith('+') and not l.startswith('+++'))
    removed = sum(1 for l in patch.splitlines() if l.startswith('-') and not l.startswith('---'))
    return added, removed


def test_imported_names(test_patch: str, pkg: str) -> set[str]:
    names = set()
    added = '\n'.join(l[1:] for l in test_patch.splitlines() if l.startswith('+'))
    for m in re.finditer(rf'from\s+{pkg}[\w.]*\s+import\s+\(?([^)\n]+(?:\n[^)]*)?)\)?', added):
        for n in re.split(r'[,\s]+', m.group(1)):
            n = n.strip().split(' as ')[0]
            if re.fullmatch(r'[A-Za-z_]\w*', n):
                names.add(n)
    return names


def snapshot_source(iid: str, pkg: str) -> str:
    chunks = []
    with tarfile.open(PROJECT / f'data/snapshots/{iid}.tgz') as tar:
        for m in tar.getmembers():
            name = m.name.lstrip('./')
            if m.isfile() and name.endswith('.py') and (name.startswith(f'{pkg}/') or name.startswith(f'src/{pkg}/')):
                chunks.append(tar.extractfile(m).read().decode('utf-8', 'replace'))
    return '\n'.join(chunks)


def run_outcome(run: str, iid: str, gold_files: list[str]) -> dict | None:
    rows = {}
    path = RESULTS / run / 'task_results.jsonl'
    if not path.exists():
        return None
    for line in open(path):
        r = json.loads(line)
        rows[r['instance_id']] = r
    if iid not in rows:
        return None
    r = rows[iid]
    patch_path = RESULTS / run / 'patches' / f'{iid}.patch'
    patch = patch_path.read_text() if patch_path.exists() else ''
    files = diff_files(patch)
    trace = json.load(open(RESULTS / run / 'traces' / f'trace_{iid}.json'))
    calls = [tc for s in trace['steps'] for tc in (s.get('tool_calls') or [])]
    edit_times = [tc['extra']['elapsed_s'] for tc in calls
                  if tc.get('function_name') in ('edit_file', 'write_file')
                  and (tc.get('extra') or {}).get('elapsed_s') is not None]
    log_path = RESULTS / run / 'test_outputs' / f'{iid}.log'
    errors = []
    if log_path.exists():
        log = log_path.read_text(errors='replace')
        errors = sorted({l.strip()[:110] for l in log.splitlines() if l.startswith('E   ') and len(l.strip()) > 4})[:2]
    return {
        'solved': r['resolved'], 'min': round(r['duration_seconds'] / 60, 1), 'calls': r['tool_calls'],
        'patch_lines': sum(diff_size(patch)), 'gold_hit': bool(set(files) & set(gold_files)),
        'files': [f for f in files if f not in gold_files][:3],
        'first_edit_min': round(min(edit_times) / 60, 1) if edit_times else None,
        'test_exit': r.get('test_exit_code'), 'errors': errors,
    }


summary = []
for iid in never:
    t = tasks[iid]
    pkg = PKG[t['repo']]
    gold_files = diff_files(t['patch'])
    ga, gr = diff_size(t['patch'])
    names = test_imported_names(t['test_patch'], pkg)
    src = snapshot_source(iid, pkg)
    missing = sorted(n for n in names if not re.search(rf'\b{re.escape(n)}\b', src))
    unmentioned = sorted(n for n in missing if n not in t['problem_statement'] + (t.get('hints_text') or ''))
    print(f'\n### {iid}  gold: +{ga}/-{gr} lines in {gold_files}')
    print(f'    statement: {len(t["problem_statement"])} chars | test imports new names: {missing or "-"}'
          f' | not in statement: {unmentioned or "-"}')
    outcomes = {}
    for run in RUNS:
        o = run_outcome(run, iid, gold_files)
        if o is None:
            continue
        outcomes[run] = o
        print(f'    {run[:18]:18} {o["min"]:>4} min {o["calls"]:>2} calls | patch {o["patch_lines"]:>4} lines'
              f' | gold file {"Y" if o["gold_hit"] else "n"} | first edit {o["first_edit_min"]} min'
              f' | exit {o["test_exit"]} | {"; ".join(o["errors"])[:150]}')
    summary.append({'iid': iid, 'gold_lines': ga + gr, 'gold_files': len(gold_files),
                    'new_names_unmentioned': unmentioned, 'new_names_missing': missing, 'runs': outcomes})

json.dump(summary, open(sys.argv[1], 'w'), indent=1)
