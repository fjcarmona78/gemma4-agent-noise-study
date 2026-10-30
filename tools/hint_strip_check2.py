import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[0]).parent))
import strip_v2 as sv  # noqa: E402
bd = sv.bd
run = 'dev91_hinted1'
scratch = Path(sys.argv[0]).with_name('hintcheck')
good = []
for l in (bd.REQUESTS / f'{run}.jsonl').read_text().splitlines():
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
for task, chains in bd.recorded_trajectories(run):
    print(f'\n== {task}: steps per chain {[len(s) for _, s in chains]}')
    for k, (messages, steps) in enumerate(chains):
        for i, (c, p, ct) in enumerate(steps):
            m = messages[c] if c < len(messages) else {}
            if sv.mentions_hint(m):
                hits = [h.group(0).replace('\n', ' ') for h in re.finditer(r'.{0,60}\b(?:hint\w*|task authors).{0,60}', json.dumps([m.get('reasoning'), m.get('content')]), re.I)]
                print(f'   chain {k} step {i} MENTIONS: {hits[:2]}')
    stripped = sv.strip_hints(chains, tok, tools)
    for k, (messages, steps) in enumerate(stripped):
        left = [m for m in messages if sv.mentions_hint(m) or (m.get('role') == 'user' and sv.HINT_WORD.search(json.dumps(m.get('content'))))]
        ids, spans, note = bd.make_example(tok, tools, messages, steps, 20480)
        print(f'   chain {k}: {len(steps)}/{len(chains[k][1])} steps kept, example {len(ids)} tokens, {len(spans)} spans, note: {note}, '
              f'hint words left in context: {len(left)}, decoded text has "hint": {bool(sv.HINT_WORD.search(tok.decode(ids)))}')
    if stripped:
        s0 = stripped[0][0]
        first = next(m for m in s0 if m.get('role') == 'assistant')
        print('   scrubbed first reasoning:', repr((first.get('reasoning') or '')[:700]))
