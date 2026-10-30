"""How different are a solved and a failed attempt at the same task? For every task that run A solved and
run B did not (or the reverse), compare the two submitted patches' changed lines.

    python tools/discordant_patches.py RUN_A RUN_B      (names under results/)

A near-identical pair (similarity >= 0.9: the same edit with a detail wrong, e.g. `self.random_state` for
`self.random_state_`) is the case where chain-level labels say least about which tokens mattered: KTO pushes
a whole chain down that was right up to that detail, the pattern behind likelihood displacement (Razin et al.
2024) and the reason for train.py's --sft-weight.
"""
import difflib
import json
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1]


def changed_lines(run: str, task: str) -> list[str]:
    f = P / 'results' / run / 'patches' / f'{task}.patch'
    if not f.exists():
        return []
    return [l for l in f.read_text(errors='ignore').splitlines()
            if l[:1] in '+-' and not l.startswith(('+++', '---'))]


def resolved(run: str) -> dict[str, bool]:
    return {r['instance_id']: bool(r['resolved'])
            for r in map(json.loads, (P / 'results' / run / 'task_results.jsonl').read_text().splitlines())}


def main():
    a, b = sys.argv[1], sys.argv[2]
    ra, rb = resolved(a), resolved(b)
    rows = []
    for t in sorted(set(ra) & set(rb)):
        if ra[t] == rb[t]:
            continue
        win, lose = (a, b) if ra[t] else (b, a)
        lw, ll = changed_lines(win, t), changed_lines(lose, t)
        sim = difflib.SequenceMatcher(None, '\n'.join(lw), '\n'.join(ll)).ratio() if lw and ll else 0.0
        rows.append((sim, t, win, len(lw), len(ll)))
    print(f'{len(rows)} discordant tasks between {a} and {b} (of {len(set(ra) & set(rb))} in both)')
    for sim, t, win, nw, nl in sorted(rows, reverse=True):
        print(f'  {sim:.2f}  {t:55s} solved by {win}; changed lines solved {nw}, failed {nl}')
    near = sum(sim >= 0.9 for sim, *_ in rows)
    print(f'near-identical (similarity >= 0.9): {near}; no patch in the failed run: '
          f'{sum(nl == 0 for *_, nl in rows)}')


if __name__ == '__main__':
    main()
