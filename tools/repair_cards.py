"""Repair-card pilot (user-approved 2026-10-24, from an external review citing STAIR, arXiv 2607.29658).

Question: does ONE compact, abstracted repair experience from ANOTHER repository, added to the issue text, help the frozen
public no-edit bundle? Arms on V2-74: relevant card (chosen by Gemma from the issue alone) vs unrelated card of similar
length (control for "the prompt merely changed") vs no card (5 existing bundle runs: 25, 26, 26, 25, 26).

Cards are WRITTEN BY GEMMA (31B, local vLLM) from SOLVED Qwen3.6-27B teacher trajectories (Apache-2.0 local model; no
Claude/GPT text). The reference patch is never used: the card is built from the teacher's own verified trajectory and diff.
Leakage guard: a task may only receive a card whose source repository differs from its own (leave-own-repo-out), because
the teacher's V2 runs are on the same 74 tasks.

    python tools/repair_cards.py make      # writes data/repair_cards/cards.jsonl (one card per solved teacher task)
    python tools/repair_cards.py select    # writes data/repair_cards/selection.jsonl (relevant + unrelated per V2-74 task)
    python tools/repair_cards.py build     # writes data/repair_cards/tasks_{relevant,unrelated}.jsonl
"""
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

P = Path(__file__).resolve().parents[1]
OUT = P / 'data/repair_cards'
URL = 'http://127.0.0.1:8000/v1/chat/completions'
MODEL = 'gemma-4-31b-it-qat-w4a16-ct'
SOURCES = [('results/dev91_teacher_qwen36', None), ('results/v2_teacher_qwen36', 'data/rebench_v2/tasks.jsonl')]
TARGET_TASKS, TARGET_IDS = 'data/rebench_v2/tasks.jsonl', 'data/rebench_v2/valid_ids.txt'
FAMILIES = ['configuration or option propagation', 'registration or dispatch of an extension, plugin or type',
            'parallel implementations that must change together', 'exception type or error-message contract',
            'serialization, parsing or format boundary', 'edge-case inputs (None, empty, unicode, negative, duplicates)',
            'API arguments, defaults or backward compatibility', 'output formatting or rendering', 'other']
CARD_PROMPT = """Below is a GitHub issue and the trajectory of an engineer who fixed it successfully: their tool calls, \
short outputs, and the final diff (the hidden tests passed).

Write a REPAIR CARD that would help a different engineer fix a SIMILAR KIND of bug in a DIFFERENT project.
Rules:
- 150 to 250 words in total.
- Do NOT name this project, its package, files, classes, functions or options. Describe them generically ("the config \
loader", "a sibling serializer", "the public entry point").
- Teach the transferable lesson: what made this fix correct, and what an engineer is likely to get wrong.
- Use exactly these headings, each on its own line:
FAMILY: <one of: {families}>
SYMPTOM: <what the issue reports, generically>
UNDERLYING CAUSE: <the real cause>
TEMPTING BUT INCOMPLETE FIX: <the plausible fix that would not pass the tests, and why>
DECISIVE OBSERVATION: <the check or reading that reveals the real cause>
CORRECTION PATTERN: <the general shape of the correct change; a tiny generic code sketch is allowed>
VERIFICATION: <how to confirm the fix before submitting>

--- ISSUE ---
{issue}
--- TRAJECTORY ---
{trajectory}
--- FINAL DIFF ---
{diff}
--- END ---

Write only the card."""
SELECT_PROMPT = """A software engineer must fix the GitHub issue below. You can give them ONE repair card describing a \
lesson from a similar past fix in another project. Pick the card whose underlying cause and correction pattern most \
likely apply to this issue.

--- ISSUE ({repo}) ---
{issue}
--- CARDS ---
{cards}
--- END ---

Answer with the card ID only, for example: C07"""


def chat(prompt, max_tokens=6000, with_reasoning=False):
    body = json.dumps({'model': MODEL, 'temperature': 0, 'max_tokens': max_tokens,
                       'messages': [{'role': 'user', 'content': prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as r:
        msg = json.loads(r.read())['choices'][0]['message']
    content = (msg.get('content') or '').strip()
    return (content, str(msg.get('reasoning_content') or msg.get('reasoning') or '')) if with_reasoning else content


def short(x, n):
    x = ' '.join(str(x).split())
    return x if len(x) <= n else x[:n] + ' ...'


def digest(trace_path, budget=7000):
    lines = []
    for s in json.loads(trace_path.read_text())['steps']:
        if s.get('source') != 'agent':
            continue
        calls = s.get('tool_calls') or []
        if isinstance(calls, str):
            continue
        for c in calls:
            lines.append(f"- {c.get('function_name')} {short(json.dumps(c.get('arguments')), 300)}")
        obs = s.get('observation')
        if obs:
            lines.append(f"  -> {short(obs if isinstance(obs, str) else json.dumps(obs), 300)}")
    text = '\n'.join(lines)
    return text if len(text) <= budget else text[:budget // 2] + '\n[...]\n' + text[-budget // 2:]


def load_tasks(*paths):
    tasks = {}
    for p in paths:
        for l in open(P / p):
            t = json.loads(l)
            tasks.setdefault(t['instance_id'], t)
    return tasks


def make():
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / 'cards.jsonl'
    done = {json.loads(l)['source_task'] for l in out.open()} if out.exists() else set()
    tasks = load_tasks(*[p for _, p in SOURCES if p], 'data/rebench_v2/tasks.jsonl', 'data/tasks.jsonl')  # dev91 = data/tasks.jsonl
    for f in P.glob('data/*/tasks.jsonl'):  # dev91 statements live in the competition task files
        for l in open(f):
            t = json.loads(l)
            tasks.setdefault(t['instance_id'], t)
    for run, _ in SOURCES:
        for r in map(json.loads, open(P / run / 'task_results.jsonl')):
            tid = r['instance_id']
            if not r['resolved'] or tid in done:
                continue
            trace, patch = P / run / 'traces' / f'trace_{tid}.json', P / run / 'patches' / f'{tid}.patch'
            issue = (tasks.get(tid) or {}).get('problem_statement')
            if not (trace.exists() and patch.exists() and issue):
                print('skip (missing trace/patch/issue):', tid, flush=True)
                continue
            t0 = time.time()
            card = chat(CARD_PROMPT.format(families='; '.join(FAMILIES), issue=issue[:5000], trajectory=digest(trace),
                                           diff=patch.read_text()[:4000]))
            fam = re.search(r'FAMILY:\s*(.+)', card)
            fam = fam.group(1).strip().lower() if fam else 'other'
            fam = next((f for f in FAMILIES if f.split()[0] in fam), 'other')
            pkg = r['repo'].split('/')[-1].lower()
            row = dict(source_task=tid, source_repo=r['repo'], family=fam, words=len(card.split()),
                       mentions_pkg=pkg in card.lower(), card=card)
            with out.open('a') as f:
                f.write(json.dumps(row) + '\n')
            print(f"{tid[:45]:45s} {fam[:30]:30s} {row['words']:4d} words pkg_leak={row['mentions_pkg']} "
                  f"{time.time() - t0:.0f}s", flush=True)


def card_body(c):
    return '\n'.join(l for l in c['card'].splitlines() if not l.startswith('FAMILY:')).strip()


def select():
    cards = [json.loads(l) for l in open(OUT / 'cards.jsonl')]
    cards = [c for c in cards if 100 <= c['words'] <= 330 and not c['mentions_pkg']]
    for i, c in enumerate(cards):
        c['id'] = f'C{i:02d}'
    print('usable cards:', len(cards))
    ids = open(P / TARGET_IDS).read().split()
    tasks = load_tasks(TARGET_TASKS)
    out = OUT / 'selection.jsonl'
    # v2 (2026-10-24 14:40): the first pass parsed only 16 of 74 answers (thinking used the 3,000-token budget before
    # the answer, so those tasks fell back to a random card). Keep the parsed rows, redo the rest with a 12k budget.
    old = [json.loads(l) for l in out.open()] if out.exists() else []
    keep = [r for r in old if r['parsed']]
    with out.open('w') as f:
        for r in keep:
            f.write(json.dumps(r) + '\n')
    done = {r['instance_id'] for r in keep}
    rng = random.Random(0)
    for tid in ids:
        if tid in done:
            continue
        t = tasks[tid]
        cand = [c for c in cards if c['source_repo'].lower() != t['repo'].lower()]
        listing = '\n'.join(
            f"{c['id']} [{c['family']}] " + short(re.search(r'SYMPTOM:\s*(.+)', c['card']).group(1) if 'SYMPTOM:' in c['card'] else c['card'], 220)
            + ' | cause: ' + short(re.search(r'UNDERLYING CAUSE:\s*(.+)', c['card']).group(1) if 'UNDERLYING CAUSE:' in c['card'] else '', 160)
            for c in cand)
        ans, reasoning = chat(SELECT_PROMPT.format(repo=t['repo'], issue=t['problem_statement'][:5000], cards=listing), 12000, True)
        ids_seen = re.findall(r'C\d{2}', ans) or re.findall(r'C\d{2}', reasoning)[-1:]  # answer first; else the last id the reasoning settled on
        m = re.match(r'C\d{2}', ids_seen[0]) if ids_seen else None
        rel = next((c for c in cand if m and c['id'] == m.group(0)), None) or rng.choice(cand)
        others = [c for c in cand if c['family'] != rel['family'] and c['source_repo'] != rel['source_repo']]
        close = [c for c in others if abs(c['words'] - rel['words']) <= 0.3 * rel['words']] or others
        unrel = rng.choice(close)
        row = dict(instance_id=tid, repo=t['repo'], relevant=rel['id'], relevant_family=rel['family'],
                   relevant_source=rel['source_task'], parsed=bool(m), unrelated=unrel['id'],
                   unrelated_family=unrel['family'], unrelated_source=unrel['source_task'],
                   answer=ans[:200], from_reasoning=bool(m) and not re.search(r'C\d{2}', ans))
        with out.open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(f"{tid[:45]:45s} -> {rel['id']} {rel['family'][:28]:28s} (parsed={row['parsed']}) | control {unrel['id']}",
              flush=True)
    json.dump({c['id']: c for c in cards}, open(OUT / 'cards_by_id.json', 'w'), indent=1)


NOTE = ('\n\n---\nREPAIR NOTE (a lesson from fixing a similar kind of bug in a different project; it may or may not '
        'apply here, so check it against this code before relying on it):\n')


def build():
    cards = json.load(open(OUT / 'cards_by_id.json'))
    sel = {json.loads(l)['instance_id']: json.loads(l) for l in open(OUT / 'selection.jsonl')}
    for arm in ('relevant', 'unrelated'):
        with open(OUT / f'tasks_{arm}.jsonl', 'w') as f:
            for l in open(P / TARGET_TASKS):
                t = json.loads(l)
                if t['instance_id'] in sel:
                    t['problem_statement'] = t['problem_statement'] + NOTE + card_body(cards[sel[t['instance_id']][arm]])
                f.write(json.dumps(t) + '\n')
        print('wrote', OUT / f'tasks_{arm}.jsonl')


if __name__ == '__main__':
    {'make': make, 'select': select, 'build': build}[sys.argv[1]]()
