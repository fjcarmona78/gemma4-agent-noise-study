"""Per-task comparison of two runs: solved, minutes, tool calls, generated tokens, compactions.

Usage: compare_runs.py RUN_A RUN_B   (names under results/)
"""
import ast
import json
import statistics
import sys
from pathlib import Path

P = Path.home() / 'projects' / 'gemma-agent'


def parse(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def stats(run: str, task: str):
    path = P / 'results' / run / 'traces' / f'trace_{task}.json'
    if not path.exists():
        return None
    steps = [s for s in json.loads(path.read_text()).get('steps') or [] if s.get('source') == 'agent']
    metrics = [parse(s.get('metrics')) or {} for s in steps]
    prompts = [m.get('prompt_tokens') for m in metrics if m.get('prompt_tokens')]
    return {'calls': sum(len(parse(s.get('tool_calls')) or []) for s in steps),
            'gen': sum(m.get('completion_tokens') or 0 for m in metrics),
            'comp': sum(1 for i in range(1, len(prompts)) if prompts[i] < prompts[i - 1] - 1500)}


def rows(run: str) -> dict:
    out = {}
    for line in (P / 'results' / run / 'task_results.jsonl').read_text().splitlines():
        r = json.loads(line)
        out[r['instance_id']] = r
    return out


a_name, b_name = sys.argv[1], sys.argv[2]
a, b = rows(a_name), rows(b_name)
tasks = [t for t in a if t in b]
print(f"{'task':16s} | {a_name[:22]:>22s}: ok  min calls  gen comp | {b_name[:22]:>22s}: ok  min calls  gen comp")
agg = {a_name: [], b_name: []}
for t in tasks:
    line = f'{t:16s}'
    for name, rr in ((a_name, a), (b_name, b)):
        s = stats(name, t) or {'calls': 0, 'gen': 0, 'comp': 0}
        r = rr[t]
        agg[name].append((r['resolved'], r['duration_seconds'] / 60, s['calls'], s['gen'], s['comp']))
        line += f" | {'':22s}  {'✓' if r['resolved'] else '✗'} {r['duration_seconds'] / 60:4.1f} {s['calls']:5d} {s['gen']:5d} {s['comp']:4d}"
    print(line)
for name, vals in agg.items():
    solved = sum(v[0] for v in vals)
    print(f'{name}: solved {solved}/{len(vals)}, mean min {statistics.mean(v[1] for v in vals):.2f}, '
          f'mean calls {statistics.mean(v[2] for v in vals):.1f}, mean gen tokens {statistics.mean(v[3] for v in vals):.0f}, '
          f'mean compactions {statistics.mean(v[4] for v in vals):.2f}, hit time limit {sum(1 for v in vals if v[1] >= 3.9)}')
only_a = [t for t in tasks if a[t]['resolved'] and not b[t]['resolved']]
only_b = [t for t in tasks if b[t]['resolved'] and not a[t]['resolved']]
print(f'solved only by {a_name}: {only_a}')
print(f'solved only by {b_name}: {only_b}')
