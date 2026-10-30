"""Turn tools/out/task_tracker.json (tools/task_tracker.py) into the two row tables the per-task tracker dashboard
reads (claude.ai/artifact/VJKxkncqMacp3WuMvAUG8Z): one row per task, one row per task x run.

    python tools/tracker_dashboard.py OUT_DIR    -> OUT_DIR/task_summary.json, OUT_DIR/run_outcomes.json
"""
import json
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1]
def main(out: Path) -> None:
    d = json.loads((P / 'tools/out/task_tracker.json').read_text())
    tasks, runs = [], []
    for t in d['tasks']:
        st = t['stages']
        tasks.append({
            'task': t['task'], 'repo': t['repo'], 'runs': t['n'], 'solved': t['solved'],
            'solve_rate': round(t['solved'] / t['n'], 3) if t['n'] else 0.0,
            'no_patch': st.get('no_patch', 0), 'wrong_file': st.get('wrong_file', 0),
            'right_file_test_fails': st.get('right_file_test_fails', 0),
            'time_capped': sum(1 for r in t['runs'] if r.get('capped')),
            'dominant_failure': t['top_failure'] or '', 'dominant_share': round(t['top_failure_share'] or 0.0, 3),
            'fix_files': ', '.join(t['gold_files']), 'issue': (t.get('statement') or '')[:400],
        })
        for r in t['runs']:
            runs.append({
                'task': t['task'], 'run': r['run'], 'stage': r['stage'], 'failing_test': r.get('test') or '',
                'assertion': r.get('error') or '', 'minutes': r.get('mins'), 'tool_calls': r.get('calls'),
                'edited_files': ', '.join(r.get('edited') or []),
            })
    out.mkdir(parents=True, exist_ok=True)
    (out / 'task_summary.json').write_text(json.dumps(tasks))
    (out / 'run_outcomes.json').write_text(json.dumps(runs))
    print(f'{len(tasks)} tasks, {len(runs)} task-run rows, {len(d["family"])} runs in the family')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
