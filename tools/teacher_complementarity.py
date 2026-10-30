"""Does a teacher run add new behavior? Per task, compare the teacher with Gemma's existing runs on the same tasks.

    python tools/teacher_complementarity.py TEACHER_RUN [GEMMA_RUN ...]

Buckets (the user's reading rule, 2026-10-13): teacher-only successes matter most where Gemma consistently missed
(0 of its runs); for shared successes, compare diagnosis (calls before the first successful edit), verification
(checks run after the last edit) and efficiency (minutes, calls) against Gemma's successful runs on the same task.
"""
import ast
import json
import statistics
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1] / 'results'
GEMMA = ['dev91_v3_rec2', 'dev91_v3_t4_rec', 'step1C_v3_dev91_t4', 'dev91_v3_t5', 'dev91_v3_t5_rep', 'dev91_v4b', 'dev91_v4c']


def field(v):
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    return v


def rows(run):
    f = P / run / 'task_results.jsonl'
    return {json.loads(l)['instance_id']: json.loads(l) for l in open(f)} if f.exists() else {}


def profile(run, task):
    """Calls before the first successful source edit, and check commands run after the last successful edit."""
    try:
        t = json.load(open(P / run / 'traces' / f'trace_{task}.json'))
    except (FileNotFoundError, ValueError):
        return None
    n, first_edit, last_edit, checks_after = 0, None, None, 0
    for s in t.get('steps', []):
        if s.get('source') != 'agent':
            continue
        for c in field(s.get('tool_calls')) or []:
            if not isinstance(c, dict):
                continue
            n += 1
            name, args = c.get('function_name'), c.get('arguments') or {}
            obs = str(field(s.get('observation')) or '')
            ok = '"status": "ok"' in obs or "'status': 'ok'" in obs
            if name in ('edit_file', 'write_file') and ok and not str(args.get('filepath', '')).startswith('/tmp'):
                first_edit = first_edit or n
                last_edit, checks_after = n, 0
            elif name == 'run_command' and last_edit and any(k in str(args.get('command', '')) for k in ('pytest', 'python3 /tmp', 'python /tmp', 'python3 -c', 'python -c')):
                checks_after += 1
    return {'calls': n, 'first_edit': first_edit, 'checks_after_last_edit': checks_after}


def main():
    teacher = sys.argv[1]
    gemma = sys.argv[2:] or GEMMA
    T = rows(teacher)
    G = {r: rows(r) for r in gemma}
    buckets = {k: [] for k in ('teacher only, Gemma never', 'teacher, Gemma sometimes', 'both always',
                               'Gemma solves, teacher failed', 'neither')}
    for task, tr in T.items():
        runs = [G[r][task] for r in gemma if task in G[r]]
        g, n = sum(x['resolved'] for x in runs), len(runs)
        if tr['resolved']:
            key = 'teacher only, Gemma never' if g == 0 else ('both always' if g == n else 'teacher, Gemma sometimes')
        else:
            key = 'Gemma solves, teacher failed' if g else 'neither'
        buckets[key].append((task, g, n))
    print(f'{teacher}: {sum(r["resolved"] for r in T.values())}/{len(T)} solved; Gemma reference = {len(gemma)} runs')
    for k, v in buckets.items():
        detail = ', '.join(f'{t} ({g}/{n})' for t, g, n in v[:12]) if k != 'neither' else ''
        print(f'  {k:30s} {len(v):3d}  {detail}')
    # shared successes: teacher vs Gemma's successful runs on the same task
    cmp = []
    for task, tr in T.items():
        if not tr['resolved']:
            continue
        g_ok = [(r, G[r][task]) for r in gemma if task in G[r] and G[r][task]['resolved']]
        if not g_ok:
            continue
        tp = profile(teacher, task)
        gp = [p for p in (profile(r, task) for r, _ in g_ok) if p]
        if not tp or not gp:
            continue
        cmp.append({
            'task': task,
            'teacher_min': tr['duration_seconds'] / 60, 'gemma_min': statistics.mean(x['duration_seconds'] for _, x in g_ok) / 60,
            'teacher_calls': tp['calls'], 'gemma_calls': statistics.mean(p['calls'] for p in gp),
            'teacher_first_edit': tp['first_edit'] or 0, 'gemma_first_edit': statistics.mean(p['first_edit'] or 0 for p in gp),
            'teacher_checks': tp['checks_after_last_edit'], 'gemma_checks': statistics.mean(p['checks_after_last_edit'] for p in gp),
        })
    if cmp:
        m = lambda k: statistics.mean(c[k] for c in cmp)
        print(f'\nshared successes ({len(cmp)} tasks): teacher vs Gemma (mean of Gemma\'s successful runs)')
        print(f"  minutes            {m('teacher_min'):5.2f} vs {m('gemma_min'):5.2f}")
        print(f"  tool calls         {m('teacher_calls'):5.1f} vs {m('gemma_calls'):5.1f}")
        print(f"  first good edit at {m('teacher_first_edit'):5.1f} vs {m('gemma_first_edit'):5.1f}  (call number; diagnosis speed)")
        print(f"  checks after edit  {m('teacher_checks'):5.2f} vs {m('gemma_checks'):5.2f}  (verification)")
    out = Path(__file__).with_name(f'complementarity_{teacher}.json')
    json.dump({'buckets': buckets, 'shared': cmp}, open(out, 'w'), indent=1)
    print(f'\nwritten {out}')


if __name__ == '__main__':
    main()
