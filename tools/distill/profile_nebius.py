"""Profile resolved Nebius OpenHands trajectories against v4's limits (28 tool calls, edit by call 12)."""
import collections
import json
import statistics
import sys

import pyarrow.parquet as pq

def iter_rows(path):
    f = pq.ParquetFile(path)
    for batch in f.iter_batches(batch_size=500, columns=['instance_id', 'repo', 'trajectory', 'resolved', 'exit_status']):
        yield from batch.to_pylist()


rows = iter_rows(sys.argv[1])
stats = collections.Counter()
n_calls, first_edit, text_rate, kinds = [], [], [], collections.Counter()
keep = 0
for r in rows:
    if r['resolved'] != 1:
        continue
    calls, edit_at, with_text, assistant_turns, bad = 0, None, 0, 0, None
    for m in r['trajectory']:
        if m.get('role') != 'assistant':
            continue
        assistant_turns += 1
        tcs = m.get('tool_calls') or []
        if (m.get('content') or '').strip():
            with_text += 1
        for tc in tcs:
            fn = tc['function']['name'] if isinstance(tc, dict) and 'function' in tc else tc.get('name')
            try:
                args = json.loads(tc['function']['arguments']) if isinstance(tc['function'].get('arguments'), str) else tc['function'].get('arguments') or {}
            except Exception:
                args = {}
            kind = fn if fn != 'str_replace_editor' else f"editor.{args.get('command')}"
            if fn == 'execute_bash' and str(args.get('is_input')).lower() == 'true':
                kind = 'execute_bash.is_input'
            kinds[kind] += 1
            if kind in ('editor.insert', 'editor.undo_edit', 'execute_bash.is_input'):
                bad = kind
            if fn in ('think', 'task_tracker'):
                continue
            calls += 1
            if kind in ('editor.str_replace', 'editor.create') and edit_at is None:
                edit_at = calls
        if len(tcs) > 1:
            stats['multi_call_turns'] += 1
    n_calls.append(calls)
    first_edit.append(edit_at or 999)
    text_rate.append(with_text / max(assistant_turns, 1))
    stats['bad:' + bad if bad else 'clean'] += 1
    if not bad and calls <= 28 and (edit_at or 999) <= 12:
        keep += 1
q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
print('tool calls per resolved run (excl think/task_tracker): median', q(n_calls, .5), 'p25', q(n_calls, .25), 'p75', q(n_calls, .75))
print('<=28 calls:', sum(c <= 28 for c in n_calls), '| first edit by call 12:', sum(e <= 12 for e in first_edit))
print('assistant turns with text before the call: median share', round(statistics.median(text_rate), 2))
print('run flags:', dict(stats))
print('tool kinds:', kinds.most_common(14))
print('KEEP (clean, <=28 calls, edit by 12):', keep)
