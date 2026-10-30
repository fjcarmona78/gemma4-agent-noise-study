"""Per-task failure tracker (2026-10-20). For every recent-set task and every run of the time-safe agent family, record
where the run failed: no patch, wrong file (no overlap with the reference fix's source files), right file but hidden test
fails (with the first failing test and assertion), time-cap, or solved. Writes tools/out/task_tracker.json and prints a
summary. Reference patches/tests are read for DIAGNOSIS ONLY — never fed to an agent.

    python tools/task_tracker.py
"""
import json, re, sys
from pathlib import Path
from collections import Counter

P = Path(__file__).resolve().parents[1]
# Default: the recent set. Another set: task_tracker.py TASKS_JSONL IDS_FILE OUT_NAME RESULTS_PREFIX
# (e.g. data/rebench_v2/tasks.jsonl data/rebench_v2/valid_ids.txt task_tracker_v2t74 v2t74_)
ARGS = sys.argv[1:] or ['data/rebench_recent/tasks.jsonl', 'data/rebench_recent/eval_ids.txt', 'task_tracker', 'recent_']
TASKS = {json.loads(l)['instance_id']: json.loads(l) for l in open(P / ARGS[0])}
IDS = [l.strip() for l in open(P / ARGS[1]) if l.strip()]
OUT_NAME, PREFIX = ARGS[2], ARGS[3]
# the time-safe family (7-min cap, 28 calls) plus its variants; older configs are excluded so rates reflect today's agent
FAMILY = sorted(r.name for r in (P / 'results').iterdir() if r.is_dir() and r.name.startswith((PREFIX + 'public_bav2_noedit_t7', PREFIX + 'v4c_pinned_cap28_noedit_safe')) and (r / 'task_results.jsonl').exists())
SRC = re.compile(r'^\+\+\+ b/(\S+)', re.M)

def src_files(patch):
    return {f for f in SRC.findall(patch or '') if f.endswith('.py') and not re.search(r'(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|conftest\.py$', f)}

def first_failure(log):
    tests = re.findall(r'^_{3,} (\S[^\n]*?) _{3,}$', log, re.M)
    err = re.search(r'^E\s+(\S[^\n]{0,160})', log, re.M)
    if not err:
        err = re.search(r'^(FAILED [^\n]{0,160}|ERROR [^\n]{0,160}|\w*Error: [^\n]{0,160})', log, re.M)
    return (tests[0] if tests else None), (err.group(1).strip() if err else None)

def norm(sig):  # collapse addresses/numbers so identical failures group together
    return re.sub(r'0x[0-9a-f]+|\d+', '#', sig) if sig else sig

rows = []
for tid in IDS:
    t = TASKS[tid]; gold = src_files(t['patch'])
    tests_new = 'new file mode' in (t['test_patch'] or '')
    runs = []
    for run in FAMILY:
        res = next((json.loads(l) for l in open(P / 'results' / run / 'task_results.jsonl') if json.loads(l)['instance_id'] == tid), None)
        if not res: continue
        pf = P / 'results' / run / 'patches' / f'{tid}.patch'; patch = pf.read_text() if pf.exists() else ''
        lf = P / 'results' / run / 'test_outputs' / f'{tid}.log'; log = lf.read_text() if lf.exists() else ''
        edited = src_files(patch)
        if res['resolved']: stage = 'solved'
        elif not res.get('agent_patch_size'): stage = 'no_patch'
        elif gold and not (edited & gold): stage = 'wrong_file'
        else: stage = 'right_file_test_fails'
        capped = res['duration_seconds'] >= 5.5 * 60 and not res['resolved']
        test, err = first_failure(log) if stage != 'solved' else (None, None)
        runs.append(dict(run=run.replace(PREFIX, '', 1), stage=stage, capped=capped, edited=sorted(edited), test=test, error=err,
                         mins=round(res['duration_seconds'] / 60, 1), calls=res.get('tool_calls')))
    solved = sum(r['stage'] == 'solved' for r in runs)
    stages = Counter(r['stage'] for r in runs)
    sigs = Counter(norm(r['error']) for r in runs if r['error'])
    top_sig, top_n = (sigs.most_common(1)[0] if sigs else (None, 0))
    rows.append(dict(task=tid, repo=t['repo'], statement=t['problem_statement'][:600], gold_files=sorted(gold),
                     hidden_tests_new_file=tests_new, runs=runs, n=len(runs), solved=solved, stages=dict(stages),
                     top_failure=top_sig, top_failure_share=round(top_n / max(1, len(runs) - solved), 2)))
out = P / 'tools/out'; out.mkdir(exist_ok=True)
(out / f'{OUT_NAME}.json').write_text(json.dumps(dict(family=FAMILY, tasks=rows), indent=1))
rows.sort(key=lambda r: r['solved'] / max(1, r['n']))
print(f'{len(FAMILY)} runs of the time-safe family; per task: solve rate, failure stages, dominant failure')
for r in rows:
    s = r['stages']
    print(f"{r['solved']:2d}/{r['n']:2d}  {r['task'][:40]:40s} nopatch={s.get('no_patch',0):2d} wrongfile={s.get('wrong_file',0):2d} rightfile={s.get('right_file_test_fails',0):2d} | {r['top_failure_share']:.0%} same: {(r['top_failure'] or '-')[:70]}")
