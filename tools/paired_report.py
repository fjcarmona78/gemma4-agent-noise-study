"""Paired comparison of two runs on the same tasks (the user's reading rule for jobs 80/81, 2026-10-13):
A-only solves, B-only solves, shared solves, and runtime; plus an exact two-sided sign test on the discordant
tasks (McNemar), so a small advantage is reported as such before anyone picks a prompt from it.

    python tools/paired_report.py RUN_A RUN_B
"""
import json
import statistics
import sys
from math import comb
from pathlib import Path

P = Path(__file__).resolve().parents[1] / 'results'


def rows(run):
    return {json.loads(l)['instance_id']: json.loads(l) for l in open(P / run / 'task_results.jsonl')}


def sign_test(a_only, b_only):
    """Two-sided exact binomial p-value for the discordant pairs (p = 0.5)."""
    n, k = a_only + b_only, min(a_only, b_only)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def describe(name, rs):
    timeouts = sum('timeout' in str(r.get('error')).lower() for r in rs)
    errors = sum(bool(r.get('error')) and 'timeout' not in str(r.get('error')).lower() for r in rs)
    return (f"{name}: {sum(r['resolved'] for r in rs)}/{len(rs)} solved | mean {statistics.mean(r['duration_seconds'] for r in rs) / 60:.2f} min, "
            f"{statistics.mean(r['tool_calls'] for r in rs):.1f} calls | empty patches {sum(r['agent_patch_size'] == 0 for r in rs)}, "
            f"timeouts {timeouts}, other errors {errors}")


def main():
    a_name, b_name = sys.argv[1], sys.argv[2]
    A, B = rows(a_name), rows(b_name)
    tasks = sorted(set(A) & set(B))
    missing = sorted(set(A) ^ set(B))
    a_only = [t for t in tasks if A[t]['resolved'] and not B[t]['resolved']]
    b_only = [t for t in tasks if B[t]['resolved'] and not A[t]['resolved']]
    shared = [t for t in tasks if A[t]['resolved'] and B[t]['resolved']]
    print(f'{len(tasks)} paired tasks' + (f' ({len(missing)} not in both runs: {missing[:5]})' if missing else ''))
    print(describe(a_name, [A[t] for t in tasks]))
    print(describe(b_name, [B[t] for t in tasks]))
    print(f'{a_name}-only solves ({len(a_only)}): {", ".join(a_only)}')
    print(f'{b_name}-only solves ({len(b_only)}): {", ".join(b_only)}')
    print(f'shared solves: {len(shared)}; neither: {len(tasks) - len(a_only) - len(b_only) - len(shared)}')
    if shared:
        m = lambda R: statistics.mean(R[t]['duration_seconds'] for t in shared) / 60
        c = lambda R: statistics.mean(R[t]['tool_calls'] for t in shared)
        print(f'on shared solves: {a_name} {m(A):.2f} min / {c(A):.1f} calls vs {b_name} {m(B):.2f} min / {c(B):.1f} calls')
    print(f'sign test on the {len(a_only) + len(b_only)} discordant tasks: p = {sign_test(len(a_only), len(b_only)):.2f} (two-sided)')


if __name__ == '__main__':
    main()
