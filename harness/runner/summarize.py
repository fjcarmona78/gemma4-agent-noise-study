"""Rebuild results/SUMMARY.md from every results/*/task_results.jsonl.

Each experiment is compared with its reference run on exactly the tasks both attempted, and
the tasks it gained or lost are listed, since aggregate scores on <=45 tasks are noisy.
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / 'results'

# Environment checks, smoke tests, and runs from before the sandbox fixes (invalid).
SKIP = {'baseline', 'smoke', 'wheelcheck', 'richcheck', 'envcheck', 'envcheck30', 'envcheck30b',
        'envcheck_dev', 'baseline_10', 'baseline_10_v2'}
# Reference run for each experiment-name prefix.
REFERENCES = {'dev45_': 'step1C_v3_dev91_t4', 'dev91_': 'step1C_v3_dev91_t4',
              'eval30_': 'step1B_v3_eval30'}
# Runs split across directories after an interruption.
MERGED = {'eval30_v2': ['eval30_v2', 'eval30_v2_part2']}


def load(dirs: list[str]) -> dict[str, dict]:
    rows = {}
    for d in dirs:
        path = RESULTS / d / 'task_results.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    rows[row['instance_id']] = row
    return rows


def describe(rows: dict[str, dict]) -> list[str]:
    n = len(rows)
    solved = sum(r['resolved'] for r in rows.values())
    by_repo = Counter()
    totals = Counter()
    for r in rows.values():
        repo = r['repo'].split('/')[-1]
        totals[repo] += 1
        by_repo[repo] += r['resolved']
    repos = ', '.join(f'{k} {by_repo[k]}/{totals[k]}' for k in sorted(totals))
    minutes = sum(r['duration_seconds'] for r in rows.values()) / 60 / n
    tools = sum(r['tool_calls'] for r in rows.values()) / n
    return [f'{solved}/{n} ({solved / n:.0%})', repos, f'{minutes:.1f} min', f'{tools:.0f}']


def main() -> None:
    runs = {}
    merged_parts = {p for parts in MERGED.values() for p in parts}
    for path in sorted(RESULTS.glob('*/task_results.jsonl')):
        name = path.parent.name
        if name in SKIP or name in merged_parts:
            continue
        runs[name] = load([name])
    for name, parts in MERGED.items():
        runs[name] = load(parts)
    runs = {k: v for k, v in runs.items() if v and any(r['tool_calls'] for r in v.values())}

    lines = ['# Evaluation summary', '', f'Updated {datetime.now():%Y-%m-%d %H:%M}.', '',
             '| Run | Solved | By repo | Avg time | Avg calls |', '|---|---|---|---|---|']
    for name in sorted(runs):
        lines.append('| ' + ' | '.join([name, *describe(runs[name])]) + ' |')

    lines += ['', '## Compared with reference (same tasks only)', '']
    for name, rows in sorted(runs.items()):
        ref_name = next((ref for prefix, ref in REFERENCES.items() if name.startswith(prefix)), None)
        if not ref_name or ref_name not in runs or ref_name == name:
            continue
        ref = runs[ref_name]
        common = sorted(set(rows) & set(ref))
        if not common:
            continue
        a = sum(ref[t]['resolved'] for t in common)
        b = sum(rows[t]['resolved'] for t in common)
        gained = [t for t in common if rows[t]['resolved'] and not ref[t]['resolved']]
        lost = [t for t in common if ref[t]['resolved'] and not rows[t]['resolved']]
        lines.append(f'- **{name}** vs {ref_name} on {len(common)} tasks: {b} vs {a} '
                     f'({b - a:+d}); gained {len(gained)} {gained}; lost {len(lost)} {lost}')

    (RESULTS / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
