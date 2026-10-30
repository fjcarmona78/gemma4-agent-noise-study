"""Rebuild every table and paired test of the paper from the released CSV files (no raw runs needed).

    python analysis/reproduce_tables.py [--results results]

Prints, in paper order: the task sets, the byte-identical Kaggle draws, the per-set run scores of the two designs
with their spread, the pooled per-task comparison and sign test, the always / coin-flip / never split, the
intervention table re-derived from the listed runs, and the failure anatomy. Only the standard library is used,
so the same file runs unchanged inside a Kaggle notebook.
"""
import argparse, csv, statistics
from collections import defaultdict
from math import comb, sqrt
from pathlib import Path


def sign_test(a, b):
    """Two-sided exact binomial p-value for a vs b discordant pairs under p = 0.5."""
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(path):
    return list(csv.DictReader(open(path, newline='')))


def table(headers, rows):
    print('| ' + ' | '.join(headers) + ' |'); print('|' + '---|' * len(headers))
    for r in rows:
        print('| ' + ' | '.join(str(x) for x in r) + ' |')
    print()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--results', default='results'); a = ap.parse_args()
    R = Path(a.results)
    tasks, runs, att, rates, kag, inter = (load(R / f) for f in
        ('tasks.csv', 'runs.csv', 'task_attempts.csv', 'task_solve_rates.csv', 'kaggle_submissions.csv', 'interventions.csv'))
    SETS = ['V2-74', 'recent-33', 'OOD-45', 'fresh-15', 'rebench-dev']
    design_runs = [r for r in runs if r['design'] in ('bundle', 'safe') and r['complete'] == '1']
    design_att = [x for x in att if x['design'] in ('bundle', 'safe')]
    by_run = defaultdict(list)
    for x in att:
        by_run[x['run_name']].append(x)

    # ---- 1. task sets --------------------------------------------------------------------------------------------
    print('## Task sets\n')
    n_in = defaultdict(int)
    for t in tasks:
        for s in t['sets'].split('|'):
            n_in[s] += 1
    table(['Set', 'Tasks', 'Source'], [(s, n_in[s], next(t['source'] for t in tasks if s in t['sets'].split('|'))) for s in SETS])
    print(f'Distinct tasks: {len(tasks)} (tasks in more than one set: {sum("|" in t["sets"] for t in tasks)})\n')

    # ---- 2. Kaggle noise ------------------------------------------------------------------------------------------
    print('## Public scores of byte-identical archives\n')
    by_sha = defaultdict(list)
    for k in kag:
        if k['sha256_prefix'] and k['public_score']:
            by_sha[(k['archive'], k['sha256_prefix'])].append(k['public_score'])
    table(['Archive', 'sha256 prefix', 'Public scores of the same file'],
          [(a_, s, ', '.join(v)) for (a_, s), v in by_sha.items() if len(v) > 1])
    p = 0.15; n = 60; sd = sqrt(n * p * (1 - p))
    print(f'Binomial model: a true rate of {p} on {n} public tasks gives SD {sd:.1f} tasks = {sd / n:.3f} in score.\n')

    # ---- 3. per-set runs of the two designs -----------------------------------------------------------------------
    print('## Runs of the two designs per set\n')
    rows = []
    for s in SETS:
        line = [s, n_in[s]]
        for d in ('bundle', 'safe'):
            sc = [int(r['n_solved']) for r in sorted(design_runs, key=lambda r: int(r['run_index'])) if r['set'] == s and r['design'] == d]
            line.append(', '.join(map(str, sc)) + (f'  (mean {statistics.mean(sc):.1f}, sd {statistics.stdev(sc):.2f})' if len(sc) > 1 else ''))
        rows.append(line)
    table(['Set', 'Tasks', 'Bundle runs', 'Safe-prompt runs'], rows)
    print(f'Complete design runs: {len(design_runs)}; attempts: {len(design_att)}; failed attempts: {sum(x["resolved"] == "0" for x in design_att)}\n')
    # independent-coin spread for the bundle on V2-74
    for d in ('bundle', 'safe'):
        pr = [float(r[f'{d}_rate']) for r in rates if 'V2-74' in r['sets'].split('|') and r[f'{d}_rate'] != '']
        print(f'Independent-coin SD on V2-74 for {d}: {sqrt(sum(q * (1 - q) for q in pr)):.2f} tasks')
    print()

    # ---- 4. pooled per-task comparison ----------------------------------------------------------------------------
    print('## Pooled per-task comparison, 216 tasks\n')
    hb = sum(float(r['bundle_rate']) > float(r['safe_rate']) for r in rates if r['bundle_rate'] and r['safe_rate'])
    hs = sum(float(r['safe_rate']) > float(r['bundle_rate']) for r in rates if r['bundle_rate'] and r['safe_rate'])
    print(f'Solve rate higher under the bundle on {hb} tasks, under the safe prompt on {hs}; sign test p = {sign_test(hb, hs):.2f}')
    # first-run pairs, duplicates once
    first = {}
    for x in design_att:
        if x['run_index'] == '1':
            first.setdefault(x['instance_id'], {})[x['design']] = first.get(x['instance_id'], {}).get(x['design']) or x['resolved']
    bo = sum(1 for v in first.values() if v.get('bundle') == '1' and v.get('safe') == '0')
    so = sum(1 for v in first.values() if v.get('safe') == '1' and v.get('bundle') == '0')
    tb = sum(v.get('bundle') == '1' for v in first.values()); ts = sum(v.get('safe') == '1' for v in first.values())
    print(f'First-run pairs: bundle {tb}, safe {ts} solved of {len(first)}; discordant {bo} vs {so}, p = {sign_test(bo, so):.2f}\n')
    cls = defaultdict(int)
    for r in rates:
        cls[r['pooled_class']] += 1
    print(f"Pooled solve rate: always (> 0.9) {cls['always']}, never (< 0.1) {cls['never']}, coin flips {cls['coin_flip']}\n")

    # ---- 5. interventions -----------------------------------------------------------------------------------------
    print('## Interventions (paper table, with the solves re-derived from the listed runs)\n')
    def solves(names):
        out = []
        for nm in names.split(';'):
            if not nm:
                continue
            if nm.endswith('*'):
                ms = sorted(k for k in by_run if k.startswith(nm[:-1]) and by_run[k][0]['design'] in ('bundle', 'safe'))
                out.append(nm + '=' + ','.join(str(sum(x['resolved'] == '1' for x in by_run[k])) for k in ms))
            elif nm in by_run:
                out.append(f"{nm}={sum(x['resolved'] == '1' for x in by_run[nm])}/{len(by_run[nm])}")
            else:
                out.append(nm + '=?')
        return '; '.join(out)
    table(['Intervention', 'Measured on', 'Result', 'Verdict', 'Runs (solves)'],
          [(i['intervention'], i['measured_on'], i['result'], i['verdict'], solves(i['runs'])) for i in inter])
    # the one significant harm: edit deadline at call 8 vs the bundle's recent-set runs
    e8 = [x for nm in ('recent_public_bav2_noedit_t7_edit8', 'recent_public_bav2_noedit_t7_edit8_rep') for x in by_run.get(nm, [])]
    base = [x for x in design_att if x['set'] == 'recent-33' and x['design'] == 'bundle']
    if e8 and base:
        rb = defaultdict(list); re_ = defaultdict(list)
        for x in base: rb[x['instance_id']].append(int(x['resolved']))
        for x in e8: re_[x['instance_id']].append(int(x['resolved']))
        up = sum(statistics.mean(re_[t]) > statistics.mean(rb[t]) for t in re_); dn = sum(statistics.mean(re_[t]) < statistics.mean(rb[t]) for t in re_)
        print(f'Edit deadline at call 8 vs the bundle on recent-33, per-task rates: higher on {up}, lower on {dn}, p = {sign_test(up, dn):.2f}\n')

    # ---- 6. failure anatomy --------------------------------------------------------------------------------------
    print('## Failure anatomy, failed attempts of both designs\n')
    failed = [x for x in design_att if x['resolved'] == '0']
    cnt = defaultdict(int)
    for x in failed:
        cnt[x['failure_class']] += 1
    label = {'only_new_hidden_tests_fail': 'Only new or changed hidden tests failed', 'no_patch': 'The agent produced no patch at all',
             'only_existing_tests_fail': 'Only tests the repository already had failed', 'both_fail': 'Both kinds failed',
             'tests_did_not_load': 'The tests did not load', 'no_test_log': 'No test log'}
    table(['What went wrong', 'Attempts', 'Share'], [(label.get(k, k), v, f'{v / len(failed):.0%}') for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])])
    nop = [x for x in failed if x['failure_class'] == 'no_patch']
    at_cap = sum(x['ended_by'] in ('tool_cap', 'time_cap') for x in nop)
    print(f'No-patch attempts that ended at the tool-call or time cap: {at_cap} of {len(nop)}')
    bundle_nop = {x['instance_id'] for x in nop if x['design'] == 'bundle'}
    print(f'Tasks where the bundle ever ended with no patch: {len(bundle_nop)} of {len(rates)}')
    # test reading within tasks that were both solved and failed (patch-producing attempts only)
    per = defaultdict(list)
    for x in design_att:
        if x['patch_bytes'] != '0' and x['read_test_file'] != '':
            per[x['instance_id']].append((x['resolved'] == '1', x['read_test_file'] == '1'))
    mixed = {t: v for t, v in per.items() if any(ok for ok, _ in v) and not all(ok for ok, _ in v)}
    ws = [sum(f for ok, f in v if ok) / sum(ok for ok, _ in v) for v in mixed.values()]
    ls = [sum(f for ok, f in v if not ok) / sum(not ok for ok, _ in v) for v in mixed.values()]
    print(f'Within {len(mixed)} tasks with both outcomes: solved runs read a test file {statistics.mean(ws):.0%}, failed runs {statistics.mean(ls):.0%}')
    fe = [int(x['first_edit_call']) for x in design_att if x['first_edit_call']]
    fs = [int(x['first_edit_call']) for x in design_att if x['first_edit_call'] and x['resolved'] == '1']
    ff = [int(x['first_edit_call']) for x in design_att if x['first_edit_call'] and x['resolved'] == '0']
    print(f'First edit call, median: solved {statistics.median(fs):.0f}, failed {statistics.median(ff):.0f} (attempts with an edit: {len(fe)})')


if __name__ == '__main__':
    main()
