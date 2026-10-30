"""Blinded patch chooser (2026-10-21, external review idea 1). For tasks that have both a passing and a failing
saved patch (from runs of the two 0.18 policies), show a fresh Gemma the issue and the two patches in random order,
no labels, and ask which is more likely to pass the hidden tests. Accuracy vs 50% says whether a two-attempt +
chooser architecture has anything to select. Diff-only (no execution) is the floor; a version with one behavioural
check is the next step if this one clears chance.

    python tools/patch_chooser.py [--per-task 4] [--seed 0]   -> tools/out/chooser.json
"""
import argparse, json, random, re, sys, time, urllib.request
from pathlib import Path

P = Path(__file__).resolve().parents[1]
RUNS = ['recent_public_bav2_noedit_t7', 'recent_public_bav2_noedit_t7_rep', 'recent_public_bav2_noedit_t7_rep2',
        'recent_public_bav2_noedit_t7_rep3', 'recent_public_bav2_noedit_t7_rep4',
        'recent_v4c_pinned_cap28_noedit_safe', 'recent_v4c_pinned_cap28_noedit_safe_rep', 'recent_v4c_pinned_cap28_noedit_safe_rep2']
TASKS = {json.loads(l)['instance_id']: json.loads(l) for l in open(P / 'data/rebench_recent/tasks.jsonl')}
URL = 'http://127.0.0.1:8000/v1/chat/completions'; MODEL = 'gemma-4-31b-it-qat-w4a16-ct'

PROMPT = """You are reviewing two candidate patches for a GitHub issue. Hidden tests, written from the issue text, will be run against the repository after one patch is applied. Decide which patch is more likely to make those tests pass. Consider: does the patch implement exactly the behaviour the issue describes, in the place where that behaviour is produced, following the repository's conventions, without breaking neighbouring behaviour?

# Issue
{issue}

# Patch 1
```diff
{p1}
```

# Patch 2
```diff
{p2}
```

Reply with exactly one line of JSON: {{"choice": 1 or 2, "reason": "<one sentence>"}}"""

def ask(issue, p1, p2):
    body = json.dumps({'model': MODEL, 'temperature': 0, 'max_tokens': 9000,
                       'messages': [{'role': 'user', 'content': PROMPT.format(issue=issue[:4000], p1=p1[:6000], p2=p2[:6000])}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        msg = json.load(r)['choices'][0]['message']
        txt = (msg.get('content') or '') + '\n' + str(msg.get('reasoning_content') or msg.get('reasoning') or '')
    # the visible answer first; if the reply ran out of tokens inside its reasoning, take the last choice stated there
    m = re.search(r'"choice"\s*:\s*([12])', msg.get('content') or '') or (re.findall(r'"choice"\s*:\s*([12])', txt) or [None])
    if isinstance(m, re.Match): choice = int(m.group(1))
    elif isinstance(m, list): choice = int(m[-1]) if m and m[-1] else None
    else: choice = None
    return choice, txt[-300:]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--per-task', type=int, default=4); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); rnd = random.Random(a.seed)
    res = {r: {json.loads(l)['instance_id']: json.loads(l) for l in open(P / 'results' / r / 'task_results.jsonl')} for r in RUNS}
    pairs = []
    for t in TASKS:
        if t not in res[RUNS[0]]: continue
        ok = [r for r in RUNS if res[r][t]['resolved'] and (P / 'results' / r / 'patches' / f'{t}.patch').exists()]
        bad = [r for r in RUNS if not res[r][t]['resolved'] and res[r][t].get('agent_patch_size') and (P / 'results' / r / 'patches' / f'{t}.patch').exists()]
        combos = [(g, b) for g in ok for b in bad]; rnd.shuffle(combos)
        pairs += [(t, g, b) for g, b in combos[:a.per_task]]
    print(f'{len(pairs)} pairs from {len({t for t,_,_ in pairs})} tasks', flush=True)
    out = []; correct = 0; answered = 0
    for i, (t, g, b) in enumerate(pairs, 1):
        pg = (P / 'results' / g / 'patches' / f'{t}.patch').read_text(); pb = (P / 'results' / b / 'patches' / f'{t}.patch').read_text()
        first_is_pass = rnd.random() < 0.5
        p1, p2 = (pg, pb) if first_is_pass else (pb, pg)
        t0 = time.time()
        try:
            choice, tail = ask(TASKS[t]['problem_statement'], p1, p2)
        except Exception as e:
            choice, tail = None, f'ERROR {e}'
        right = choice is not None and ((choice == 1) == first_is_pass)
        answered += choice is not None; correct += right
        out.append(dict(task=t, pass_run=g, fail_run=b, pass_shown_as=1 if first_is_pass else 2, choice=choice, correct=right, secs=round(time.time()-t0,1), tail=tail))
        print(f'{i:3d}/{len(pairs)} {t[:36]:36s} pass@{1 if first_is_pass else 2} chose {choice} {"OK " if right else "no "} ({time.time()-t0:.0f}s)  running acc {correct}/{answered}', flush=True)
    (P / 'tools/out').mkdir(exist_ok=True); (P / 'tools/out/chooser.json').write_text(json.dumps(out, indent=1))
    by = {}
    for o in out: by.setdefault(o['task'], []).append(o['correct'])
    print('\nSUMMARY accuracy', f'{correct}/{answered}', f'= {correct/max(1,answered):.0%}', '(chance 50%)')
    for t, v in by.items(): print(f'  {t[:42]:42s} {sum(v)}/{len(v)}')
    same = [o for o in out if o['pass_run'].split('_rep')[0].split('_safe')[0] == o['fail_run'].split('_rep')[0].split('_safe')[0]]
    if same: print(f'  same-policy pairs (deployable scenario): {sum(o["correct"] for o in same)}/{len(same)}')

if __name__ == '__main__':
    main()
