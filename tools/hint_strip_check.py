"""Dry-run strip_hints on the hinted run's successes so far: how much of each trajectory survives."""
import json
import re
import shutil
import sys
from pathlib import Path

P = Path.home() / 'projects' / 'gemma-agent'
sys.path.insert(0, str(P / 'finetune'))
import build_dataset as bd  # noqa: E402

run = 'dev91_hinted1'
scratch = Path(sys.argv[0]).with_name('hintcheck')
# copy the live request log, keeping only complete lines (the proxy is still appending)
lines = (bd.REQUESTS / f'{run}.jsonl').read_text().splitlines()
good = []
for l in lines:
    try:
        json.loads(l)
        good.append(l)
    except ValueError:
        pass
(scratch / f'{run}.jsonl').write_text('\n'.join(good) + '\n')
bd.REQUESTS = scratch

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(str(bd.MODEL_DIR))
tools = bd.load_tools()
print('served config matches v3 probe:', bd.served_config(scratch / f'{run}.jsonl') == bd.served_config(bd.PROBE_LOG))
for task, chains in bd.recorded_trajectories(run):
    print(f'\n== {task}: {len(chains)} chain(s), steps per chain {[len(s) for _, s in chains]}, in EVAL30: {task in bd.EVAL30}')
    user = chains[0][0][1]['content']
    m = bd.HINT_RE.search(user)
    print('   hint found in user message:', bool(m), '|', (m.group(0).strip()[:160] if m else ''))
    # where does the model first mention the hint?
    for k, (messages, steps) in enumerate(chains):
        for i, (c, p, ct) in enumerate(steps):
            nxt = messages[c] if c < len(messages) else {}
            text = (nxt.get('reasoning') or '') + ' ' + str(nxt.get('content') or '')
            hit = re.search(r'.{0,100}hint.{0,100}', text, re.I | re.S)
            if hit:
                print(f'   first mention: chain {k} step {i}: ...{hit.group(0)!r}...')
                break
        else:
            continue
        break
    stripped = bd.strip_hints(chains, tok, tools)
    print('   after strip_hints: steps per chain', [len(s) for _, s in stripped])
    for k, (messages, steps) in enumerate(stripped):
        ids, spans, note = bd.make_example(tok, tools, messages, steps, 20480)
        print(f'   example chain {k}: {len(ids)} tokens, {len(spans)} steps used, note: {note}')
