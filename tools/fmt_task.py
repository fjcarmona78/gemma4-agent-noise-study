"""Print one line per newly finished task, with the reference run's result and a running tally.

Usage: fmt_task.py RESULTS_FILE START END REFERENCE_RUN_DIR...
"""
import json
import sys
from pathlib import Path

results_file, start, end = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
ref = {}
for ref_dir in sys.argv[4:]:
    path = Path(ref_dir) / 'task_results.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            ref[row['instance_id']] = row['resolved']

rows = [json.loads(l) for l in results_file.read_text().splitlines() if l.strip()]
run = results_file.parent.name
mark = lambda ok: '✓' if ok else '✗'
for i in range(start, min(end, len(rows))):
    done = rows[:i + 1]
    r = rows[i]
    solved = sum(x['resolved'] for x in done)
    ref_solved = sum(ref.get(x['instance_id'], False) for x in done)
    ref_mark = mark(ref[r['instance_id']]) if r['instance_id'] in ref else '-'
    print(f"[{run}] {i + 1}: {r['instance_id']} {mark(r['resolved'])} (ref {ref_mark}) "
          f"{r['duration_seconds'] / 60:.1f} min | tally {solved}/{i + 1} vs ref {ref_solved}", flush=True)
