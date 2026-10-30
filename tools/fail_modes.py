"""Failure modes of the no-edit bundle on the recent set: for every task, first edit call, edit count, calls, outcome."""
import json, sys, collections
from pathlib import Path
P = Path(__file__).resolve().parents[1] / 'results'
runs = sys.argv[1:]
rows = {}
for run in runs:
    res = {json.loads(l)['instance_id']: json.loads(l) for l in open(P/run/'task_results.jsonl')}
    for tf in (P/run/'traces').glob('trace_*.json'):
        tid = tf.name[6:-5]; d = json.load(open(tf)); r = res.get(tid, {})
        calls = []; 
        for s in d['steps']:
            for tc in s.get('tool_calls') or []:
                a = tc.get('arguments'); a = a if isinstance(a, dict) else {}
                cmd = str(a.get('command', ''))
                kind = 'edit' if (tc.get('function_name') == 'write_file' or 'write_text' in cmd) else \
                       'test' if 'pytest' in cmd or 'unittest' in cmd else \
                       'submit' if tc.get('function_name') == 'submit_patch' else 'read'
                ok = 'ok' if "print('ok')" in cmd and 'ok' in str(s.get('observation', {}).get('content', ''))[:200] else ''
                calls.append((kind, ok))
        first_edit = next((i+1 for i, c in enumerate(calls) if c[0] == 'edit'), None)
        n_edit = sum(1 for c in calls if c[0] == 'edit'); n_edit_ok = sum(1 for c in calls if c[0]=='edit' and c[1]=='ok')
        n_test = sum(1 for c in calls if c[0] == 'test'); submitted = any(c[0] == 'submit' for c in calls)
        rows.setdefault(tid, []).append(dict(run=run, solved=r.get('resolved'), calls=len(calls), first_edit=first_edit, edits=n_edit, edits_ok=n_edit_ok, tests=n_test, submitted=submitted, patch=r.get('agent_patch_size', 0), mins=round(r.get('duration_seconds', 0)/60, 1)))
# classify failures
cat = collections.Counter(); detail = []
for tid, rs in sorted(rows.items()):
    for x in rs:
        if x['solved']: continue
        if x['patch'] == 0 and x['edits'] == 0: c = 'no edit attempted'
        elif x['patch'] == 0: c = 'edits attempted, none applied'
        elif x['mins'] >= 5.5 and not x['submitted']: c = 'edited, hit cap (no submit)'
        elif x['tests'] == 0: c = 'edited+submitted, never tested'
        else: c = 'edited+tested+submitted, wrong fix'
        cat[c] += 1; detail.append((c, tid, x))
print('FAILURE MODES over', len(runs), 'runs:')
for c, n in cat.most_common(): print(f'  {n:3d}  {c}')
print('\nfirst-edit call index, failures vs solves:')
fe_f = [x['first_edit'] for rs in rows.values() for x in rs if not x['solved'] and x['first_edit']]
fe_s = [x['first_edit'] for rs in rows.values() for x in rs if x['solved'] and x['first_edit']]
import statistics as st
print(f'  failures: n={len(fe_f)} median={st.median(fe_f) if fe_f else None}  >12: {sum(1 for v in fe_f if v>12)}')
print(f'  solves:   n={len(fe_s)} median={st.median(fe_s) if fe_s else None}  >12: {sum(1 for v in fe_s if v>12)}')
print('\nno-edit / none-applied cases:')
for c, tid, x in detail:
    if c in ('no edit attempted', 'edits attempted, none applied'):
        print(f"  {tid:45s} {x['run'][-5:]:>5s} calls={x['calls']:2d} edits={x['edits']} ok={x['edits_ok']} tests={x['tests']} {x['mins']}min")

print('\nSOLVES: tested share and timing')
sv = [x for rs in rows.values() for x in rs if x['solved']]; fl = [x for rs in rows.values() for x in rs if not x['solved']]
print(f"  solves n={len(sv)}: never tested {sum(1 for x in sv if x['tests']==0)}, mean calls {st.mean(x['calls'] for x in sv):.1f}, mean edits {st.mean(x['edits'] for x in sv):.1f}")
print(f"  fails  n={len(fl)}: never tested {sum(1 for x in fl if x['tests']==0)}, mean calls {st.mean(x['calls'] for x in fl):.1f}, mean edits {st.mean(x['edits'] for x in fl):.1f}")
print('first-edit histogram (call index buckets 1-4,5-8,9-12,13-16,17-20,21-28,29+):')
for lab, xs in (('solves', sv), ('fails', fl)):
    b = collections.Counter(min((x['first_edit']-1)//4, 6) if x['first_edit'] else 7 for x in xs)
    print(f"  {lab:7s}", [b.get(i, 0) for i in range(7)], 'never', b.get(7, 0))
