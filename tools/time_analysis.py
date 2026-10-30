"""Where do the minutes go? Per-task time breakdown of a recorded run (proxy log + results).

For each task: model-call time (split by generated tokens: reasoning vs tool-call text), summarization
calls, and the gaps between calls (tool execution + harness overhead), by outcome group.

Usage: time_analysis.py RUN
"""
import collections
import json
import statistics
import sys
from pathlib import Path

PROJECT = Path.home() / 'projects' / 'gemma-agent'
sys.path.insert(0, str(PROJECT / 'finetune'))
import build_dataset as bd  # noqa: E402

run = sys.argv[1]
records = []
for line in (PROJECT / 'finetune' / 'requests' / f'{run}.jsonl').read_text().splitlines():
    try:
        records.append(json.loads(line))
    except ValueError:
        pass
results = bd.run_results(run)
by_user = {}
for task in results:
    trace = bd.load_trace(run, task)
    if trace and trace.get('steps'):
        by_user[next(s['message'] for s in trace['steps'] if s.get('source') == 'user')] = task

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(str(bd.MODEL_DIR))


def ntok(text: str) -> int:
    return len(tok(text, add_special_tokens=False)['input_ids']) if text else 0


rows = []
tool_gaps = collections.defaultdict(list)
for task_records in bd.split_tasks(records):
    task = by_user.get(task_records[0]['request']['messages'][1]['content'])
    if task is None:
        continue
    task_records.sort(key=lambda r: r['ts'])
    agent = [r for r in task_records if r['request'].get('tools')]
    summ = [r for r in task_records if not r['request'].get('tools')]
    llm_s = sum(r['seconds'] for r in agent)
    summ_s = sum(r['seconds'] for r in summ)
    gaps = 0.0
    for a, b in zip(task_records, task_records[1:]):
        gap = b['ts'] - (a['ts'] + a['seconds'])
        if gap > 0:
            gaps += gap
            if a['request'].get('tools'):
                calls = a['response']['choices'][0]['message'].get('tool_calls') or []
                for c in calls:
                    name = c['function']['name']
                    args = c['function']['arguments']
                    kind = name
                    if name == 'run_command':
                        kind = 'run_command:pytest' if 'pytest' in args else 'run_command:other'
                    tool_gaps[kind].append(gap)
    reason_tok = sum(ntok(r['response']['choices'][0]['message'].get('reasoning') or '') for r in agent)
    gen_tok = sum(r['response']['usage']['completion_tokens'] for r in agent)
    long_calls = [r['seconds'] for r in agent if r['seconds'] > 30]
    res = results[task]
    minutes = res['duration_seconds'] / 60
    group = 'solved' if res['resolved'] else ('timeout' if minutes >= 3.9 else 'early fail')
    window = task_records[-1]['ts'] + task_records[-1]['seconds'] - task_records[0]['ts']
    rows.append({'task': task, 'group': group, 'minutes': minutes, 'llm_s': llm_s, 'summ_s': summ_s,
                 'gaps_s': gaps, 'window_s': window, 'calls': len(agent), 'summaries': len(summ),
                 'gen_tok': gen_tok, 'reason_tok': reason_tok, 'long_calls': len(long_calls),
                 'long_calls_s': sum(long_calls)})

print(f'{run}: {len(rows)} tasks')
fmt = lambda v: f'{v:7.1f}'
for group in ('solved', 'timeout', 'early fail', 'ALL'):
    g = [r for r in rows if group == 'ALL' or r['group'] == group]
    if not g:
        continue
    mean = lambda k: statistics.mean(r[k] for r in g)
    win = mean('window_s')
    print(f"\n{group} (n={len(g)}): agent window {win:.0f} s | model calls {mean('llm_s'):.0f} s ({100 * mean('llm_s') / win:.0f}%), "
          f"summaries {mean('summ_s'):.0f} s ({100 * mean('summ_s') / win:.0f}%), tools+overhead {mean('gaps_s'):.0f} s ({100 * mean('gaps_s') / win:.0f}%)")
    print(f"   calls {mean('calls'):.1f}, summaries {mean('summaries'):.2f}, generated tokens {mean('gen_tok'):.0f} "
          f"(reasoning {mean('reason_tok'):.0f} = {100 * mean('reason_tok') / max(mean('gen_tok'), 1):.0f}%), "
          f"calls >30 s: {mean('long_calls'):.2f} per task taking {mean('long_calls_s'):.0f} s")
print('\ntool time (gap after the call) by kind: n, median s, mean s, total share of all tool time')
total = sum(sum(v) for v in tool_gaps.values())
for kind, v in sorted(tool_gaps.items(), key=lambda kv: -sum(kv[1])):
    print(f'   {kind:22s} n={len(v):5d} median {statistics.median(v):5.1f} mean {statistics.mean(v):5.1f} share {100 * sum(v) / total:4.1f}%')
json.dump(rows, open(Path(sys.argv[0]).with_name(f'time_rows_{run}.json'), 'w'), indent=1)
