"""Compare two prompts across repeated runs on the same tasks: which solves are stable, and whether the prompts
differ beyond one prompt's own run-to-run noise.

    python tools/prompt_stability.py --a RUN [RUN ...] --b RUN [RUN ...]

Per task it counts solves per prompt (0..n runs). Buckets: stable advantage (one prompt solves in every run,
the other in none), leaning (one prompt solves more often), and equal. The sign test runs on the tasks where
the per-prompt solve counts differ; the noise line shows how often the SAME prompt disagrees with itself.
"""
import argparse
import itertools
import json
from math import comb
from pathlib import Path

P = Path(__file__).resolve().parents[1] / 'results'


def rows(run):
    return {json.loads(l)['instance_id']: json.loads(l)['resolved'] for l in open(P / run / 'task_results.jsonl')}


def sign_test(x, y):
    n, k = x + y, min(x, y)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', nargs='+', required=True)
    ap.add_argument('--b', nargs='+', required=True)
    args = ap.parse_args()
    A = [rows(r) for r in args.a]
    B = [rows(r) for r in args.b]
    tasks = sorted(set.intersection(*(set(r) for r in A + B)))
    ca = {t: sum(r[t] for r in A) for t in tasks}
    cb = {t: sum(r[t] for r in B) for t in tasks}
    na, nb = len(A), len(B)
    print(f'{len(tasks)} tasks | A = {args.a} solves {[sum(r[t] for t in tasks) for r in A]} | B = {args.b} solves {[sum(r[t] for t in tasks) for r in B]}')
    print(f'mean solves per run: A {sum(ca.values()) / na:.1f}, B {sum(cb.values()) / nb:.1f}')
    stable_a = [t for t in tasks if ca[t] == na and cb[t] == 0]
    stable_b = [t for t in tasks if cb[t] == nb and ca[t] == 0]
    lean_a = [t for t in tasks if ca[t] / na > cb[t] / nb and t not in stable_a]
    lean_b = [t for t in tasks if cb[t] / nb > ca[t] / na and t not in stable_b]
    print(f'stable A-only (A every run, B never): {len(stable_a)} {stable_a}')
    print(f'stable B-only (B every run, A never): {len(stable_b)} {stable_b}')
    print(f'leaning A: {len(lean_a)} | leaning B: {len(lean_b)} | equal: {len(tasks) - len(stable_a) - len(stable_b) - len(lean_a) - len(lean_b)}')
    xa, xb = len(stable_a) + len(lean_a), len(stable_b) + len(lean_b)
    print(f'sign test on tasks where the prompts differ ({xa} vs {xb}): p = {sign_test(xa, xb):.2f}')
    for name, runs in (('A', A), ('B', B)):
        if len(runs) > 1:
            d = [sum(r1[t] != r2[t] for t in tasks) for r1, r2 in itertools.combinations(runs, 2)]
            print(f'self-disagreement of {name} between its own runs: {d} tasks')


if __name__ == '__main__':
    main()
