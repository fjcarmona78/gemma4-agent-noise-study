"""Spec-gap check (2026-10-22): for each task, find names the hidden tests use that the reference fix *introduces*
(defined or newly referenced in the gold patch's added lines, absent from its removed/context lines) and that the
problem statement never mentions. A task whose tests need such a name cannot be solved without guessing it.
Reference patches/tests are read for DIAGNOSIS ONLY, never fed to an agent.

    python tools/spec_gap.py TASKS_JSONL TRACKER_JSON
"""
import json, re, sys
from collections import Counter

IDENT = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]{3,})\b')
DEF = re.compile(r'^\+\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)|^\+\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:=]', re.M)
KW = set('self None True False return import from class def pass raise assert with async await lambda yield else elif while print kwargs args'.split())

def lines(patch, sign):
    return [l[1:] for l in (patch or '').splitlines() if l.startswith(sign) and not l.startswith(sign * 3)]

def introduced(patch):
    added = '\n'.join(lines(patch, '+'))
    old = '\n'.join(lines(patch, '-') + [l[1:] for l in (patch or '').splitlines() if l.startswith(' ')])
    old_ids = set(IDENT.findall(old))
    defs = {a or b for a, b in DEF.findall('\n'.join('+' + l for l in lines(patch, '+')))}
    return {i for i in (defs | set(IDENT.findall(added))) if i not in old_ids and i not in KW}

def main():
    tasks = {json.loads(l)['instance_id']: json.loads(l) for l in open(sys.argv[1])}
    tr = {t['task']: t for t in json.load(open(sys.argv[2]))['tasks']}
    rows = []
    for tid, t in tasks.items():
        if tid not in tr: continue
        new = introduced(t['patch'])
        test_ids = set(IDENT.findall('\n'.join(lines(t['test_patch'], '+'))))
        stmt = t['problem_statement'] or ''
        gap = sorted(i for i in new & test_ids if i not in stmt and i.lower() not in stmt.lower())
        r = tr[tid]; rate = r['solved'] / max(1, r['n'])
        rows.append((tid, r['solved'], r['n'], gap))
    cls = lambda s, n: 'always' if s == n else 'never' if s == 0 else 'sometimes'
    c = Counter(); g = Counter()
    for tid, s, n, gap in rows:
        c[cls(s, n)] += 1; g[cls(s, n)] += bool(gap)
    for k in ('never', 'sometimes', 'always'):
        print(f'{k:9s}: {c[k]:3d} tasks, {g[k]:3d} with a spec gap ({g[k] / max(1, c[k]):.0%})')
    print('\nnever-solved tasks WITHOUT a spec gap (reachable in principle):')
    for tid, s, n, gap in rows:
        if s == 0 and not gap: print('  ', tid)
    print('\nexamples of gaps (never-solved):')
    for tid, s, n, gap in [r for r in rows if r[1] == 0 and r[3]][:8]:
        print(f'   {tid[:40]:40s} {gap[:5]}')


if __name__ == '__main__':
    main()
