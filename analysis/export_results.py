"""Export the per-task results behind the paper from a local `results/` tree into the CSV files in `results/`.

This script needs the raw run directories (task_results.jsonl, traces/, test_outputs/, patches/) and the task
files (tasks.jsonl per set), which are not part of the repository. The CSV files it writes ARE in the repository,
and `analysis/reproduce_tables.py` rebuilds every table and test of the paper from them alone.

    python analysis/export_results.py --project /path/to/gemma-agent --out results

What goes out (external tasks only: SWE-rebench V1/V2 and OOD-45; nothing from the competition's own data):

  tasks.csv              one row per distinct task: instance_id, repo, created_at, sets, source
  runs.csv               one row per local run: run_name, set, design, variant, run_index, counts, caps
  task_attempts.csv      one row per (run, task): outcome, calls, minutes, how the attempt ended, failure anatomy,
                         file stage, first edit call, test reading flags
  task_solve_rates.csv   per task x design solve rates over the complete runs of the two designs, pooled class
  kaggle_submissions.csv the public leaderboard scores of our submissions (static table)
  interventions.csv      the paper's intervention table with the run names behind each row (static table)
"""
import argparse, csv, glob, json, re, statistics
from collections import defaultdict
from pathlib import Path

# ---- the five local task sets: results prefix -> (set name, id list, tasks.jsonl, source label) -----------------
SETS = {
    'v2t74_':       ('V2-74',       'data/rebench_v2/valid_ids.txt',      'data/rebench_v2/tasks.jsonl',      'SWE-rebench V2'),
    'recent_':      ('recent-33',   'data/rebench_recent/eval_ids.txt',   'data/rebench_recent/tasks.jsonl',  'SWE-rebench V2 (2025)'),
    'ood45_':       ('OOD-45',      'data/ood45/valid_ids.txt',           'data/ood45/tasks.jsonl',           'OOD-45'),
    'fresh15_':     ('fresh-15',    'data/rebench_fresh15/eval_ids.txt',  'data/rebench_fresh15/tasks.jsonl', 'SWE-rebench V2 (2025)'),
    'rebench_dev_': ('rebench-dev', 'data/rebench_scale/dev_ids.txt',     'data/rebench_scale/tasks.jsonl',   'SWE-rebench V1'),
}
# ---- the two designs of the paper (results stem after the set prefix; `_rep`, `_rep2`, ... are repeats) --------
DESIGNS = {'public_bav2_noedit_t7': 'bundle', 'v4c_pinned_cap28_noedit_safe': 'safe'}
SKIP = re.compile(r'(_gold|_empty|_probe$|_at28$|_pythonpath_bug$|_part\d$|_http400$)')
TESTFILE = re.compile(r'(^|/)(tests?/|test_[^/]*\.py$|[^/]*_test\.py$|conftest\.py)')
SRC = re.compile(r'^\+\+\+ b/(\S+)', re.M)
EDIT_CMD = re.compile(r"write_text\(|\bsed -i\b|\bcat\s*>|\btee\b|open\([^)]*['\"]w['\"]")

KAGGLE = [  # ref, submitted (UTC), archive, design, sha256 prefix, public score, note
    ('57039009', '2026-10-10', 'v3', 'early', '05302483', '0.12', 'first real prompt, 5 min / 50 calls'),
    ('57103830', '2026-10-12', 'v3 + LoRA adapter v1', 'early', '04b344dd', '0.12', 'self-distilled adapter; adapters are served'),
    ('57150376', '2026-10-13', 'v4b', 'early', '522378d3', '0.13', 'single agent, contract-first prompt'),
    ('57194829', '2026-10-14', 'v4c_pinned', 'early', '858b6d60', '0.13', 'problem statement pinned in the system prompt'),
    ('57245072', '2026-10-15', 'v4c_pinned_think2048', 'early', '', '0.12', 'thinking budget 2,048'),
    ('57272241', '2026-10-16', 'public_bav2', 'bundle', '6ff73d10', '0.18', 'the public Budget-Aware v2 bundle, unchanged'),
    ('57310976', '2026-10-17', 'v4c_pinned_cap28', 'safe', '1f0fbddd', '', 'no score: exceeded the 12-hour runtime'),
    ('57356408', '2026-10-18', 'public_bav2_noedit_t7', 'bundle', 'ddb98f1b', '0.17', 'bundle without edit_file, 7-minute cap'),
    ('57391476', '2026-10-19', 'v4c_pinned_cap28_noedit_safe', 'safe', '66d2a865', '0.18', 'the safe prompt'),
    ('57431779', '2026-10-20', 'public_bav2_noedit_t7_mintools', 'bundle', '', '0.10', 'graph tools and read_file removed'),
    ('57481110', '2026-10-21', 'v4c_pinned_cap28_noedit_safe', 'safe', '66d2a865', '0.13', 'byte-identical replicate of 57391476'),
    ('57517743', '2026-10-22', 'public_bav2_noedit_t7', 'bundle', 'ddb98f1b', '', 'no score: Kaggle-side error on rerun'),
    ('57557415', '2026-10-23', 'public_bav2_noedit_t7', 'bundle', 'ddb98f1b', '0.13', 'byte-identical replicate of 57356408'),
    ('57594720', '2026-10-24', 'v4c_pinned_cap28_noedit_safe', 'safe', '66d2a865', '0.13', 'byte-identical replicate #2 of 57391476'),
    ('57626303', '2026-10-25', 'public_bav2_noedit_t7_e4b_zero_lora', 'E4B', '', '0.03', 'Gemma 4 E4B with an all-zero rank-16 adapter'),
]

INTERVENTIONS = [  # intervention, measured on, result, verdict, run names (local), baseline run names
    ('Prompt rewrites, three generations', 'Kaggle', '0.12, 0.13, 0.13', 'Flat', '', ''),
    ('Thinking budget 4,096 / 2,048 / 1,024 tokens', 'recent-33, Kaggle', '9, 9 / 11, 11 / 9; Kaggle 0.12 at 2,048', 'Flat',
     'recent_v4c_pinned;recent_v4c_pinned_t3;recent_think2048_t3;recent_think2048_t4;recent_think1024_t3', ''),
    ('Temperature 0 instead of 0.2', 'recent-33', '11, 12 vs bundle mean 12.4', 'Flat', 'recent_public_bav2_noedit_t7_temp0;recent_public_bav2_noedit_t7_temp0_rep', 'recent_public_bav2_noedit_t7*'),
    ('Call cap 40 or 50 instead of 28', 'recent-33', '14, 9 / 14', 'Flat', 'recent_public_bav2_c40;recent_public_bav2_c40_rep;recent_public_bav2_c50', 'recent_public_bav2;recent_public_bav2_rep'),
    ('Edit deadline moved to call 8', 'recent-33', '13, 9 vs 13.0; 1 vs 9 tasks, p = 0.02', 'Harm', 'recent_public_bav2_noedit_t7_edit8;recent_public_bav2_noedit_t7_edit8_rep', 'recent_public_bav2_noedit_t7*'),
    ('Stricter verification rules', 'recent-33', '10, 13 vs 13.0', 'Flat', 'recent_public_bav2_noedit_t7_verify;recent_public_bav2_noedit_t7_verify_rep', 'recent_public_bav2_noedit_t7*'),
    ('Removing the unused graph tools', 'recent-33, Kaggle', '12, 14, 12 locally; Kaggle 0.10', 'Flat or worse',
     'recent_public_bav2_noedit_t7_mintools;recent_public_bav2_noedit_t7_mintools_rep;recent_public_bav2_noedit_t7_mintools_rep2', 'recent_public_bav2_noedit_t7*'),
    ('Leaderboard "0.24" prompt edits', 'recent-33', '10', 'Flat', 'recent_public_hh024', 'recent_public_bav2;recent_public_bav2_rep'),
    ('LoRA SFT on own successes, three adapters', 'eval sets, Kaggle', 'Kaggle 0.12, equal to no adapter', 'Flat', '', ''),
    ("LoRA SFT on a 27B teacher's successes", 'recent-33, rebench-dev', '9 vs 9; 31 vs 30 to 34', 'Flat', 'recent_lora_v3;rebench_dev_lora_v3', 'recent_v4c_pinned;rebench_dev_v4c_pinned'),
    ('KTO on own successes and failures', 'recent-33', '7, 10 vs 9', 'Flat', 'recent_kto_v1;recent_kto_v2', 'recent_v4c_pinned'),
    ('Choosing between two finished patches', 'pooled pairs', '57% correct', 'Chance', '', ''),
    ('Oracle pre-fix code in the statement', '8 persistent failures', '0 of 24', 'No effect',
     'recent_ctx8_public_bav2_noedit_t7;recent_ctx8_public_bav2_noedit_t7_rep;recent_ctx8_public_bav2_noedit_t7_rep2', ''),
    ('Five attempts at temperature 0.8', '13 never-solved tasks', '0 of 65', 'Ceiling',
     'recent_never_temp08_a1;recent_never_temp08_a2;recent_never_temp08_a3;recent_never_temp08_a4;recent_never_temp08_a5', ''),
    ('Routing tasks to the better design by features', '216 tasks', '115.1 vs 115.6 expected solves', 'No signal', '', ''),
    ('Retrieved repair note, relevant vs unrelated', 'V2-74', '27 vs 29, same-day plain run 27', 'No effect',
     'v2t74_bundle_cards_relevant;v2t74_bundle_cards_unrelated', 'v2t74_public_bav2_noedit_t7_rep5'),
    ('One fixed generic repair note', 'V2-74', '7 of 26 at stop, behind plain', 'No effect', 'v2t74_bundle_cards_generic', 'v2t74_public_bav2_noedit_t7*'),
    ('Gemma 4 E4B, same agent', 'recent-33, Kaggle', '2, 5 of 33; Kaggle 0.03', 'Far below 31B', 'recent_e4b_bundle;recent_e4b_bundle_rep', ''),
    ('E4B with a test-and-retry prompt', 'recent-33', '4, 6 vs 2, 5', 'Flat', 'recent_e4b_testloop;recent_e4b_testloop_rep', 'recent_e4b_bundle;recent_e4b_bundle_rep'),
    ('E4B distilled from 31B trajectories', 'recent-33', '5, 3 vs 2, 5', 'Flat', 'recent_e4b_distill_v1;recent_e4b_distill_v1_rep', 'recent_e4b_bundle;recent_e4b_bundle_rep'),
]


def src_files(patch):
    return {f for f in SRC.findall(patch or '') if f.endswith('.py') and not TESTFILE.search(f)}


def run_caps(project, run_name, prefix):
    """Tool-call cap and local minute cap of a run, from its submission's eval_config.yaml (local minutes = 0.8 x Kaggle)."""
    stem = run_name[len(prefix):]
    for cand in (stem, re.sub(r'_rep\d*$', '', stem)):
        f = project / 'submissions' / cand / 'eval_config.yaml'
        if f.exists():
            txt = f.read_text()
            calls = re.search(r'max_tool_calls:\s*(\d+)', txt); mins = re.search(r'max_time_minutes:\s*(\d+)', txt)
            return (int(calls.group(1)) if calls else 28), (0.8 * int(mins.group(1)) if mins else 5.6)
    m = re.search(r'_c(\d\d)\b', stem)
    return (int(m.group(1)) if m else 28), 5.6


def trace_features(path):
    """First edit call, edit/test counts and test-reading flags from the ATIF trace (heuristic, see README)."""
    out = dict(first_edit_call='', n_edits='', n_tests='', read_test_file='', n_calls='')
    if not path.exists():
        return out
    try:
        steps = json.loads(path.read_text()).get('steps', [])
    except Exception:
        return out
    calls = []
    for s in steps:
        tc = s.get('tool_calls') or []
        if isinstance(tc, str):
            continue
        for c in tc:
            a = c.get('arguments'); a = a if isinstance(a, dict) else {}
            fn = c.get('function_name') or ''
            cmd = str(a.get('command') or ''); fp = str(a.get('filepath') or a.get('path') or '')
            if fn == 'write_file' or fn == 'edit_file' or (fn == 'run_command' and EDIT_CMD.search(cmd)):
                kind = 'edit'
            elif fn == 'run_command' and re.search(r'\bpytest\b|unittest|\btox\b', cmd):
                kind = 'test'
            elif fn == 'read_file' and TESTFILE.search(fp.lstrip('./')):
                kind = 'read_test'
            else:
                kind = 'other'
            calls.append(kind)
    first = next((i + 1 for i, k in enumerate(calls) if k == 'edit'), '')
    return dict(first_edit_call=first, n_edits=sum(k == 'edit' for k in calls), n_tests=sum(k == 'test' for k in calls),
                read_test_file=int(any(k == 'read_test' for k in calls)), n_calls=len(calls))


def anatomy(row, run_dir, task):
    """Failure anatomy of a failed attempt, as in tools/gate_check.py."""
    if row['resolved']:
        return 'solved', ''
    if not row.get('agent_patch_size'):
        return 'no_patch', ''
    log = run_dir / 'test_outputs' / f"{row['instance_id']}.log"
    if not log.exists():
        return 'no_test_log', ''
    text = log.read_text(errors='ignore')
    failed = re.findall(r'^(?:FAILED|ERROR) (\S+)', text, re.M)
    if not failed:
        return 'tests_did_not_load', ''
    tp = task.get('test_patch') or ''
    touched = set(re.findall(r'def (test\w*)', tp))
    names = {re.split(r'[\[]', x.split('::')[-1])[0] for x in failed}
    untouched = {x for x in names if x not in touched and x not in tp}
    first = failed[0][:160]
    if not untouched:
        return 'only_new_hidden_tests_fail', first
    if untouched == names:
        return 'only_existing_tests_fail', first
    return 'both_fail', first


def file_stage(row, run_dir, task):
    """Where the attempt ended relative to the reference fix's source files, as in tools/task_tracker.py."""
    if row['resolved']:
        return 'solved'
    if not row.get('agent_patch_size'):
        return 'no_patch'
    pf = run_dir / 'patches' / f"{row['instance_id']}.patch"
    gold = src_files(task.get('patch'))
    edited = src_files(pf.read_text(errors='ignore')) if pf.exists() else set()
    if gold and not (edited & gold):
        return 'wrong_file'
    return 'right_file_test_fails'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', required=True, help='local project root holding results/, data/, submissions/')
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    P = Path(a.project).expanduser(); O = Path(a.out); O.mkdir(parents=True, exist_ok=True)

    # tasks per set
    set_ids, tasks, task_sets, task_source = {}, {}, defaultdict(list), {}
    for prefix, (name, idf, tjf, source) in SETS.items():
        ids = [l.strip() for l in open(P / idf) if l.strip()]
        set_ids[name] = ids
        for l in open(P / tjf):
            t = json.loads(l)
            if t['instance_id'] in ids:
                tasks.setdefault(t['instance_id'], t)
        for i in ids:
            task_sets[i].append(name); task_source.setdefault(i, source)

    # runs
    runs, attempts = [], []
    rates = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # task -> design -> [runs, solved]
    for prefix, (sname, *_ ) in SETS.items():
        n = len(set_ids[sname])
        names = sorted(d.name for d in (P / 'results').iterdir()
                       if d.is_dir() and d.name.startswith(prefix) and (d / 'task_results.jsonl').exists() and not SKIP.search(d.name))
        for run_name in names:
            stem = run_name[len(prefix):]
            base = re.sub(r'_rep\d*$', '', stem)
            design = DESIGNS.get(base, 'variant')
            variant = '' if design != 'variant' else base
            m = re.search(r'_rep(\d*)$', stem)
            run_index = 1 if not m else (2 if m.group(1) == '' else int(m.group(1)) + 1)
            rows = [json.loads(l) for l in open(P / 'results' / run_name / 'task_results.jsonl')]
            complete = len(rows) == n
            if design != 'variant' and not complete:
                continue            # an interrupted repeat of a design run (never used in the paper)
            cap_calls, cap_min = run_caps(P, run_name, prefix)
            run_dir = P / 'results' / run_name
            n_solved = empty = time_caps = 0; mins = []; calls = []
            for r in rows:
                tid = r['instance_id']; t = tasks.get(tid, {})
                dur = float(r.get('duration_seconds') or 0); tc = int(r.get('tool_calls') or 0)
                err = r.get('error') or ''
                if 'timeout' in str(err).lower() or dur >= cap_min * 60 - 15:
                    ended = 'time_cap'
                elif tc >= cap_calls:
                    ended = 'tool_cap'
                elif err:
                    ended = 'error'
                else:
                    ended = 'finished'
                anat, first_fail = anatomy(r, run_dir, t)
                tf = trace_features(run_dir / 'traces' / f'trace_{tid}.json')
                attempts.append(dict(run_name=run_name, set=sname, design=design, variant=variant, run_index=run_index,
                                     instance_id=tid, repo=t.get('repo', r.get('repo', '')), resolved=int(bool(r['resolved'])),
                                     patch_bytes=int(r.get('agent_patch_size') or 0), tool_calls=tc, llm_calls=r.get('total_llm_calls', ''),
                                     minutes=round(dur / 60, 2), ended_by=ended, error=str(err)[:120].replace('\n', ' '),
                                     failure_class=anat, file_stage=file_stage(r, run_dir, t), first_failed_test=first_fail, **tf))
                n_solved += bool(r['resolved']); empty += not r.get('agent_patch_size'); time_caps += ended == 'time_cap'
                mins.append(dur / 60); calls.append(tc)
                if design != 'variant':
                    rates[tid][design][0] += 1; rates[tid][design][1] += bool(r['resolved'])
            runs.append(dict(run_name=run_name, set=sname, design=design, variant=variant, run_index=run_index, n_tasks=len(rows),
                             n_solved=n_solved, complete=int(complete), cap_calls=cap_calls, cap_local_minutes=cap_min,
                             mean_minutes=round(statistics.mean(mins), 2), mean_calls=round(statistics.mean(calls), 1),
                             empty_patches=empty, time_caps=time_caps))

    def write(name, rows, fields):
        with open(O / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        print(f'{name}: {len(rows)} rows')

    write('tasks.csv', [dict(instance_id=i, repo=tasks[i].get('repo', ''), created_at=tasks[i].get('created_at', ''),
                             sets='|'.join(task_sets[i]), source=task_source[i], license=tasks[i].get('license', ''))
                        for i in sorted(tasks)], ['instance_id', 'repo', 'created_at', 'sets', 'source', 'license'])
    write('runs.csv', runs, list(runs[0].keys()))
    write('task_attempts.csv', attempts, list(attempts[0].keys()))
    rate_rows = []
    for tid in sorted(rates):
        d = rates[tid]; b, s = d.get('bundle', [0, 0]), d.get('safe', [0, 0])
        pooled = (b[1] + s[1]) / (b[0] + s[0])
        rate_rows.append(dict(instance_id=tid, sets='|'.join(task_sets[tid]), bundle_runs=b[0], bundle_solved=b[1],
                              bundle_rate=round(b[1] / b[0], 3) if b[0] else '', safe_runs=s[0], safe_solved=s[1],
                              safe_rate=round(s[1] / s[0], 3) if s[0] else '', pooled_rate=round(pooled, 3),
                              pooled_class='always' if pooled > 0.9 else 'never' if pooled < 0.1 else 'coin_flip'))
    write('task_solve_rates.csv', rate_rows, list(rate_rows[0].keys()))
    write('kaggle_submissions.csv', [dict(zip(['ref', 'submitted_utc', 'archive', 'design', 'sha256_prefix', 'public_score', 'note'], k)) for k in KAGGLE],
          ['ref', 'submitted_utc', 'archive', 'design', 'sha256_prefix', 'public_score', 'note'])
    write('interventions.csv', [dict(zip(['intervention', 'measured_on', 'result', 'verdict', 'runs', 'baseline_runs'], k)) for k in INTERVENTIONS],
          ['intervention', 'measured_on', 'result', 'verdict', 'runs', 'baseline_runs'])


if __name__ == '__main__':
    main()
