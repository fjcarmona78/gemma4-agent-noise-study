"""Audit a built training dataset (output of finetune/build_dataset.py): what is actually supervised.

    python tools/audit_dataset.py finetune/data/v3.jsonl [MAX_LEN]

MAX_LEN (default 12288, train.py's --max-len) is applied the way train.py loads examples: spans ending past
it are dropped, so the report shows what training actually supervises, not just what the builder wrote.

Reports, per run and source: examples, distinct tasks and repos, why examples stopped early (the
builder's note), steps supervised of steps recorded. Over all examples: supervised tool calls by name,
how many examples end in a supervised submit_patch (the finish), how many supervise a pytest/python
check after the last supervised edit (verification), and whether the task statement is present in the
example's context (it is not, after a compaction, unless the prompt pins it).
"""
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'finetune'))
import build_dataset as b  # noqa: E402

TASK_FILES = ['data/tasks.jsonl', 'data/rebench_v2/tasks.jsonl', 'data/rebench_scale/tasks.jsonl', 'data/rebench_recent/tasks.jsonl']


def statements() -> dict[str, str]:
    out = {}
    for f in TASK_FILES:
        p = b.PROJECT / f
        if p.exists():
            for line in open(p):
                t = json.loads(line)
                out.setdefault(t['instance_id'], t.get('problem_statement') or '')
    return out


def main():
    from transformers import AutoTokenizer
    path = Path(sys.argv[1])
    tok = AutoTokenizer.from_pretrained(str(b.MODEL_DIR))
    max_len = int(sys.argv[2]) if len(sys.argv) > 2 else 12288
    raw = [json.loads(l) for l in open(path)]
    ex, cut_steps, cut_examples = [], 0, 0
    for e in raw:  # as finetune/train.py load_examples
        spans = [s for s in e['spans'] if s[1] <= max_len]
        cut_steps += len(e['spans']) - len(spans)
        cut_examples += len(spans) < len(e['spans'])
        if spans:
            ex.append(dict(e, spans=spans, steps=len(spans)))
    print(f"training cap {max_len} tokens: {cut_examples} of {len(raw)} examples lose their later steps; "
          f"{cut_steps} of {sum(len(e['spans']) for e in raw)} supervised steps dropped; {len(raw) - len(ex)} examples dropped entirely")
    # What the cap removes: tool calls in the dropped spans (the review's question: does truncation take away the
    # correction, the verification or the stop decision rather than exploration?)
    lost = collections.Counter()
    lost_by_label = collections.defaultdict(collections.Counter)
    for e in raw:
        for s, t in e['spans']:
            if t > max_len:
                names = re.findall(r'<\|tool_call>call:(\w+)', tok.decode(e['input_ids'][s:t])) or ['(text answer)']
                lost.update(names)
                lost_by_label[e.get('label')].update(names)
    print(f"   tool calls in the dropped steps: {dict(lost.most_common())}")
    if len(lost_by_label) > 1:
        for label, c in lost_by_label.items():
            print(f"      label {label}: {dict(c.most_common())}")
    stmts = statements()
    repos = {run: {t: r.get('repo') for t, r in b.run_results(run).items()} for run in {e['run'] for e in ex}}
    by_run = collections.defaultdict(list)
    for e in ex:
        by_run[(e['run'], e['source'])].append(e)
    for (run, src), es in sorted(by_run.items()):
        tasks = {e['task'] for e in es}
        print(f"{run} ({src}): {len(es)} examples, {len(tasks)} tasks, {len({repos[run].get(t) for t in tasks})} repos")
        print(f"   notes: {dict(collections.Counter(e['note'] for e in es))}")
        print(f"   steps supervised {sum(e['steps'] for e in es)} of {sum(e['steps_total'] for e in es)} recorded; "
              f"{sum(len(e['input_ids']) for e in es):,} tokens")
    calls, finish, with_edit, check_after_edit, present, absent_first = collections.Counter(), 0, 0, 0, 0, 0
    task_calls = collections.defaultdict(set)  # (run, task, label) -> tool names supervised in any of its chains
    for e in ex:
        ids = e['input_ids']
        texts = [tok.decode(ids[s:t]) for s, t in e['spans']]
        names = [re.findall(r'<\|tool_call>call:(\w+)', t) for t in texts]
        for n in names:
            calls.update(n)
            task_calls[(e['run'], e['task'], e.get('label'))].update(n)
        if any('submit_patch' in n for n in names):  # the closing summary after the submit is usually the last span
            finish += 1
        edits = [i for i, n in enumerate(names) if any(x in ('edit_file', 'write_file') for x in n)]
        if edits:
            with_edit += 1
            if any(re.search(r'call:run_command\{[^}]*(pytest|python)', t) for t in texts[edits[-1] + 1:]):
                check_after_edit += 1
        head = tok.decode(ids[:min(len(ids), 8000)])
        st = (stmts.get(e['task']) or '').strip()[:60]
        if st and st in head:
            present += 1
        elif e['chain'] == 0:
            absent_first += 1
    print(f"supervised tool calls: {dict(calls.most_common())}")
    print(f"examples with a supervised submit_patch (the finish): {finish}/{len(ex)}")
    print(f"examples with a supervised edit: {with_edit}; of those, a supervised pytest/python check after the last edit: {check_after_edit}")
    print(f"task statement present in the context: {present}/{len(ex)} (missing in {absent_first} first-chain examples)")
    # Per task run: does any chain supervise the finish? The builder's own truncation (note 'truncated at N tokens')
    # removes steps before the training cap sees them, so this, not the cap line above, shows a lost finish.
    if any(label is not None for *_, label in task_calls):
        for label in (True, False):
            runs = [c for (_, _, l), c in task_calls.items() if l == label]
            edit = sum(bool(c & {'edit_file', 'write_file'}) for c in runs)
            print(f"{'solved' if label else 'failed'} task runs: {len(runs)}; with a supervised submit_patch "
                  f"{sum('submit_patch' in c for c in runs)}, a supervised edit {edit}, neither {sum(not (c & {'edit_file', 'write_file', 'submit_patch'}) for c in runs)}")


if __name__ == '__main__':
    main()
