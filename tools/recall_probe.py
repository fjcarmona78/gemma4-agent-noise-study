"""Recall probe (2026-10-22, user-approved): does Gemma remember how maintainers fixed an issue, i.e. can it supply
the information the hidden tests need that the statement lacks?

For each task the local model gets only what the agent gets on Kaggle (repository name + problem statement; no PR
number, no repository access, no tools) and is asked for its best recollection of the upstream fix, labelled REMEMBERED or GUESSING.
Scoring (reference patches are read for DIAGNOSIS ONLY, never shown to the model):
  G = the reference fix's distinctive new material that the statement does NOT contain: string literals of 10+ chars
      and introduced identifiers that look like code (underscore, CamelCase or digit, 6+ chars).
  hit = the answer contains at least one item of G.
Controls: the recent set (created Feb-Jul 2025, after the model's training cutoff, so recall is impossible and hits
measure guessing), and a mismatch score (each answer against another task's G) for generic-word hits.

    python tools/recall_probe.py OUT_JSONL TASKS_JSONL:IDS_FILE:LABEL [...]
"""
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spec_gap import introduced  # noqa: E402

URL = 'http://127.0.0.1:8000/v1/chat/completions'
MODEL = 'gemma-4-31b-it-qat-w4a16-ct'
PROMPT = """This is a GitHub issue from the repository {repo}.

--- ISSUE ---
{statement}
--- END ---

Do you remember how the maintainers of {repo} actually fixed this issue upstream? If you recall the real fix, describe \
it as concretely as you can: the files and functions changed, any new or renamed functions, classes, parameters or \
constants (exact names), the exact text of any new error or warning messages, and the behaviour the new tests check. \
Give your best recollection even if you are unsure. Start your answer with exactly one word on its own line: \
REMEMBERED if you specifically recall this upstream fix, or GUESSING if you are reconstructing a plausible fix."""
LIT = re.compile(r'''(["'])((?:(?!\1).){10,200})\1''')
CODEY = re.compile(r'^(?=.{6,})(?:.*_.*|.*[a-z][A-Z].*|.*\d.*)$')


def gold_items(task):
    stmt = task['problem_statement'] or ''
    added = '\n'.join(l[1:] for l in (task['patch'] or '').splitlines() if l.startswith('+') and not l.startswith('+++'))
    removed = '\n'.join(l[1:] for l in (task['patch'] or '').splitlines() if l.startswith(('-', ' ')) and not l.startswith('---'))
    lits = {m.group(2).strip() for m in LIT.finditer(added)}
    lits = {s for s in lits if s not in removed and s.lower() not in stmt.lower() and len(s.strip()) >= 10}
    names = {n for n in introduced(task['patch']) if CODEY.match(n) and n not in stmt}
    return sorted(lits), sorted(names)


def hit(answer, lits, names):
    a_low = answer.lower()
    got_l = [s for s in lits if s.lower() in a_low]
    got_n = [n for n in names if re.search(r'\b' + re.escape(n) + r'\b', answer)]
    return got_l, got_n


def ask(repo, statement):
    body = json.dumps({'model': MODEL, 'temperature': 0, 'max_tokens': 6000, 'messages': [
        {'role': 'user', 'content': PROMPT.format(repo=repo, statement=statement[:6000])}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as r:
        msg = json.loads(r.read())['choices'][0]['message']
    return (msg.get('content') or '').strip(), str(msg.get('reasoning_content') or msg.get('reasoning') or '')


def main():
    out = Path(sys.argv[1])
    done = {json.loads(l)['instance_id'] for l in out.open()} if out.exists() else set()
    for spec in sys.argv[2:]:
        tasks_path, ids_path, label = spec.split(':')
        ids = set(Path(ids_path).read_text().split())
        for t in map(json.loads, Path(tasks_path).read_text().splitlines()):
            if t['instance_id'] not in ids or t['instance_id'] in done:
                continue
            lits, names = gold_items(t)
            t0 = time.time()
            try:
                answer, reasoning = ask(t['repo'], t['problem_statement'] or '')
            except Exception as e:  # keep going; a failed call is recorded, not retried
                answer, reasoning = f'ERROR {e}', ''
            got_l, got_n = hit(answer, lits, names)
            row = dict(instance_id=t['instance_id'], set=label, created=str(t.get('created_at', ''))[:10],
                       gold_lits=lits, gold_names=names, unknown=not answer.strip().upper().startswith('REMEMBERED'),
                       hit_lits=got_l, hit_names=got_n, answer=answer, reasoning_chars=len(reasoning),
                       secs=round(time.time() - t0, 1))
            with out.open('a') as f:
                f.write(json.dumps(row) + '\n')
            print(f"{label:6s} {t['instance_id'][:45]:45s} unknown={row['unknown']!s:5s} items={len(lits) + len(names):2d} "
                  f"hits={len(got_l) + len(got_n)} {row['secs']}s", flush=True)
    summarize(out)


def summarize(out):
    rows = [json.loads(l) for l in out.open()]
    random.seed(0)
    for label in sorted({r['set'] for r in rows}):
        rs = [r for r in rows if r['set'] == label and not r['answer'].startswith('ERROR')]
        scorable = [r for r in rs if r['gold_lits'] or r['gold_names']]
        hits = [r for r in scorable if r['hit_lits'] or r['hit_names']]
        # mismatch control: each answer scored against another task's gold items from the same set
        mm = 0
        for r in scorable:
            o = random.choice([x for x in scorable if x['instance_id'] != r['instance_id']])
            gl, gn = hit(r['answer'], o['gold_lits'], o['gold_names'])
            mm += bool(gl or gn)
        print(f'{label}: {len(rs)} tasks, said REMEMBERED {sum(not r["unknown"] for r in rs)}; scorable {len(scorable)}; '
              f'HIT {len(hits)} ({len(hits) / max(1, len(scorable)):.0%}); mismatch-control hits {mm} '
              f'({mm / max(1, len(scorable)):.0%})')
        for r in hits:
            print(f'   hit {r["instance_id"][:45]:45s} {r["created"]} {(r["hit_lits"] + r["hit_names"])[:4]}')


if __name__ == '__main__':
    if sys.argv[1:2] == ['--summary']:
        summarize(Path(sys.argv[2]))
    else:
        main()
