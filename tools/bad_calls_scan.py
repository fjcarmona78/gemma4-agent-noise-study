"""Count trained steps whose tool call does not match the tool schema (unknown/missing argument names),
over the trajectories kept in a built dataset."""
import json
import sys
from collections import Counter
from pathlib import Path

P = Path.home() / 'projects' / 'gemma-agent'
sys.path.insert(0, str(P / 'finetune'))
import build_dataset as bd  # noqa: E402

bd.REQUESTS = Path(sys.argv[1])
kept = {(r['task'], r['run']) for r in map(json.loads, open(sys.argv[2]))}
tools = bd.load_tools()
schema = {t['function']['name']: (set(t['function']['parameters'].get('properties', {})),
                                  set(t['function']['parameters'].get('required', []))) for t in tools}


def bad(msg):
    for call in msg.get('tool_calls') or []:
        name = call['function']['name']
        args = call['function'].get('arguments')
        try:
            args = args if isinstance(args, dict) else json.loads(args or '{}')
        except ValueError:
            return f'{name}: unparseable arguments'
        if name not in schema:
            return f'unknown tool {name}'
        props, req = schema[name]
        extra, missing = set(args) - props, req - set(args)
        if extra or missing:
            return f'{name}: extra {sorted(extra)[:2]} missing {sorted(missing)}'
    return None


stats, examples = Counter(), []
sources = [(run, bd.recorded_trajectories) for run in bd.RECORDED_RUNS if (bd.REQUESTS / f'{run}.jsonl').exists()]
sources += [(run, bd.trace_trajectories) for run in bd.TRACE_RUNS]
for run, fn in sources:
    try:
        trajectories = list(fn(run))
    except FileNotFoundError:
        continue
    for task, chains in trajectories:
        if (task, run) not in kept:
            continue
        for messages, steps in chains:
            for c, p, ct in steps:
                msg = messages[c] if c < len(messages) else {}
                stats['steps'] += 1
                why = bad(msg)
                if why:
                    stats['bad'] += 1
                    examples.append((run, task, why))
print(dict(stats))
for e in examples[:25]:
    print('  ', e)
print('by kind:', Counter(e[2].split(':')[0] for e in examples).most_common())
