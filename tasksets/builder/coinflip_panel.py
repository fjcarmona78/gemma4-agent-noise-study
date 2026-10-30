"""Coin-flip panel (2026-10-24, user-approved brainstorm item 2).

Across every complete run of the two 0.18-family archives (public no-edit bundle, safe prompt) on the five local sets,
a task is a COIN FLIP when its pooled solve rate is between 10% and 90%. Always-solved and never-solved tasks add run
time but no information about an intervention, so a candidate is tested on the coin-flip tasks only: fewer GPU hours per
run, more runs per decision.

Writes data/coinflip_panel/:
  <set>_ids.txt       panel tasks per local set (each set has its own task file / images; one run_recorded per set)
  panel.json          per task: set, pooled rate, bundle rate, safe rate, number of runs
  report.json         panel size, expected bundle solves per panel run, run-to-run SD, minimum detectable gain

Caveat: the panel is selected on the same runs that give the baseline rates, so those rates are slightly pulled toward
the middle (regression to the mean). A fresh control run of the bundle on the panel, made alongside the candidate's runs,
removes that bias; the report gives both comparisons' power.

    python tools/coinflip_panel.py [LOW HIGH]
"""
import glob
import json
import math
import random
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1]
SETS = {'v2t74': 74, 'recent': 33, 'ood45': 45, 'fresh15': 15, 'rebench_dev': 53}
POL = {'bundle': 'public_bav2_noedit_t7', 'safe': 'v4c_pinned_cap28_noedit_safe'}
OUT = P / 'data' / 'coinflip_panel'


def complete_runs():
    """(set, policy, run name, {task: resolved}) for every complete run."""
    for s, n in SETS.items():
        for pol, stem in POL.items():
            dirs = glob.glob(str(P / 'results' / f'{s}_{stem}')) + glob.glob(str(P / 'results' / f'{s}_{stem}_rep*'))
            for d in sorted(dirs):
                f = Path(d, 'task_results.jsonl')
                if not f.is_file():
                    continue
                rows = [json.loads(l) for l in open(f)]
                if len(rows) == n:
                    yield s, pol, Path(d).name, {r['instance_id']: bool(r['resolved']) for r in rows}


def main():
    low, high = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) == 3 else (0.1, 0.9)
    runs = list(complete_runs())
    outcomes = {}  # task -> {'set': s, 'bundle': [..], 'safe': [..]}
    for s, pol, _, res in runs:
        for t, ok in res.items():
            outcomes.setdefault(t, {'set': s, 'bundle': [], 'safe': []})[pol].append(ok)
    rate = lambda xs: sum(xs) / len(xs) if xs else None
    panel = {}
    for t, o in outcomes.items():
        pooled = rate(o['bundle'] + o['safe'])
        if low <= pooled <= high:
            panel[t] = {'set': o['set'], 'pooled': round(pooled, 3), 'bundle': rate(o['bundle']), 'safe': rate(o['safe']),
                        'bundle_runs': len(o['bundle']), 'safe_runs': len(o['safe'])}
    OUT.mkdir(parents=True, exist_ok=True)
    for s in SETS:
        ids = sorted(t for t, v in panel.items() if v['set'] == s)
        (OUT / f'{s}_ids.txt').write_text('\n'.join(ids) + ('\n' if ids else ''))
    json.dump(panel, open(OUT / 'panel.json', 'w'), indent=1)

    # Run-to-run spread of the bundle on the panel, from its actual runs (per set, summed with independent draws).
    expected = sum(v['bundle'] for v in panel.values())
    var = sum(v['bundle'] * (1 - v['bundle']) for v in panel.values())  # Bernoulli per task given its rate
    sd_run = math.sqrt(var)
    # Simulated power: candidate k runs with every task's rate raised by `gain` tasks in total (spread evenly), vs
    # (a) the historical bundle rates (fixed) or (b) one fresh control run per candidate run.
    random.seed(0)
    tasks = list(panel.values())

    def detect(gain_tasks, k, fresh_control, sims=4000):
        add = gain_tasks / len(tasks)
        hits = 0
        for _ in range(sims):
            cand = sum(sum(random.random() < min(1, v['bundle'] + add) for v in tasks) for _ in range(k)) / k
            if fresh_control:
                ctrl = sum(sum(random.random() < v['bundle'] for v in tasks) for _ in range(k)) / k
                sd = math.sqrt(2 * var / k)
            else:
                ctrl, sd = expected, math.sqrt(var / k)
            hits += (cand - ctrl) / sd > 1.645  # one-sided 5%
        return round(hits / sims, 2)

    by_set = {s: sum(v['set'] == s for v in panel.values()) for s in SETS}
    minutes = 0.0
    for s, pol, name, _ in runs:
        if pol == 'bundle':
            rows = [json.loads(l) for l in open(P / 'results' / name / 'task_results.jsonl')]
            minutes += sum(r['duration_seconds'] for r in rows if r['instance_id'] in panel) / 60
    bundle_task_runs = sum(v['bundle_runs'] for v in panel.values())
    report = {
        'band': [low, high], 'panel_tasks': len(panel), 'by_set': by_set, 'tasks_total': len(outcomes),
        'runs_used': {'bundle': sum(p == 'bundle' for _, p, _, _ in runs), 'safe': sum(p == 'safe' for _, p, _, _ in runs)},
        'bundle_expected_solves_per_panel_run': round(expected, 2), 'bundle_run_sd': round(sd_run, 2),
        'est_gpu_hours_per_panel_run': round(minutes / bundle_task_runs * len(panel) / 60, 2),
        'power_vs_history': {f'+{g} tasks, {k} runs': detect(g, k, False) for g in (2, 3, 4) for k in (1, 2, 3)},
        'power_vs_fresh_control': {f'+{g} tasks, {k} runs each': detect(g, k, True) for g in (2, 3, 4) for k in (1, 2, 3)},
    }
    json.dump(report, open(OUT / 'report.json', 'w'), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
