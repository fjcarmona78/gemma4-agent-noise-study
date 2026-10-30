"""Compact digest of one task's agent attempts for manual failure review (2026-10-22): the statement, the reference
fix and the failing hidden test (DIAGNOSIS ONLY, never fed to an agent), then per run each tool call (name, short
arguments, short result) with the agent's reasoning trimmed, and the final patch.

    python tools/trace_digest.py TASKS_JSONL TASK_ID RUN [RUN ...] > digest.md
"""
import json
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1]
CUT = 260


def short(x, n=CUT):
    x = ' '.join(str(x).split())
    return x if len(x) <= n else x[:n] + ' …'


def main():
    tasks_path, tid, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
    t = next(json.loads(l) for l in open(P / tasks_path) if json.loads(l)['instance_id'] == tid)
    print(f'# {tid}\n\n## Statement\n{t["problem_statement"][:2500]}\n\n## Reference fix\n```diff\n{t["patch"][:3000]}\n```')
    for run in runs:
        R = P / 'results' / run
        res = next((json.loads(l) for l in open(R / 'task_results.jsonl') if json.loads(l)['instance_id'] == tid), {})
        print(f'\n## Run {run}: resolved={res.get("resolved")} calls={res.get("tool_calls")} '
              f'min={round((res.get("duration_seconds") or 0) / 60, 1)}')
        tr = R / 'traces' / f'trace_{tid}.json'
        if tr.exists():
            for s in json.loads(tr.read_text())['steps']:
                if s.get('source') != 'agent':
                    continue
                calls = s.get('tool_calls') or []
                if isinstance(calls, str):
                    print(f'- think: {short(s.get("message"), 200)} | calls: {short(calls)}')
                    continue
                for c in calls:
                    print(f'- **{c.get("function_name")}** {short(json.dumps(c.get("arguments")), 200)}')
                obs = s.get('observation')
                if obs:
                    print(f'  → {short(obs if isinstance(obs, str) else json.dumps(obs))}')
                if s.get('message'):
                    print(f'  ~ {short(s["message"], 180)}')
        pf = R / 'patches' / f'{tid}.patch'
        print(f'\n### Final patch\n```diff\n{pf.read_text()[:2500] if pf.exists() else "(none)"}\n```')
        lf = R / 'test_outputs' / f'{tid}.log'
        if lf.exists():
            log = lf.read_text(errors='ignore')
            errs = [l for l in log.splitlines() if l.startswith(('E ', 'FAILED'))][:6]
            print('### Hidden-test failure\n```\n' + '\n'.join(errs) + '\n```')


if __name__ == '__main__':
    main()
