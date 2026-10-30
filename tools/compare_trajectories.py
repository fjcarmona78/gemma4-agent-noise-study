"""Side-by-side tool-call sequences of two recorded runs on the same tasks, to find the first consequential
difference between a run that solved a task and one that did not (e.g. the teacher's ten V2-only wins).

    python tools/compare_trajectories.py RUN_A RUN_B [TASK ...]
Without TASK arguments: every task RUN_A solved and RUN_B did not.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'finetune'))
import build_dataset as b  # noqa: E402


def calls(messages: list[dict]) -> list[str]:
    """Compact one-line description of every tool call, in order, with the tool result's first line."""
    out, pending = [], {}
    for m in messages:
        if m.get('role') == 'assistant':
            for tc in m.get('tool_calls') or []:
                f = tc['function']
                try:
                    a = json.loads(f['arguments']) if isinstance(f['arguments'], str) else f['arguments']
                except Exception:
                    a = {'raw': str(f['arguments'])[:80]}
                if f['name'] == 'run_command':
                    s = f"run_command: {str(a.get('command', ''))[:110]}"
                elif f['name'] in ('read_file',):
                    s = f"read_file: {a.get('path') or a.get('file_path')} {a.get('start_line', '')}-{a.get('end_line', '')}"
                elif f['name'] in ('edit_file', 'write_file'):
                    s = f"{f['name']}: {a.get('path') or a.get('file_path')} ({len(str(a.get('old_string', '')))}->{len(str(a.get('new_string', a.get('content', ''))))} chars)"
                else:
                    s = f"{f['name']}: {json.dumps(a)[:60]}"
                pending[tc.get('id')] = len(out)
                out.append(s)
        elif m.get('role') == 'tool':
            i = pending.get(m.get('tool_call_id'))
            c = m.get('content')
            if isinstance(c, list):
                c = ' '.join(x.get('text', '') for x in c if isinstance(x, dict))
            first = (c or '').strip().splitlines()[:1]
            if i is not None and first:
                out[i] += f"  => {first[0][:70]}"
    return out


def main():
    a, bb = sys.argv[1], sys.argv[2]
    ra, rb = b.run_results(a), b.run_results(bb)
    tasks = sys.argv[3:] or sorted(t for t in ra if ra[t]['resolved'] and t in rb and not rb[t]['resolved'])
    ta = {t: chains for t, chains in b.recorded_trajectories(a, include_failed=True)}
    tb = {t: chains for t, chains in b.recorded_trajectories(bb, include_failed=True)}
    for t in tasks:
        print(f"\n===== {t}  ({a}: {'solved' if ra[t]['resolved'] else 'failed'}, {bb}: {'solved' if rb[t]['resolved'] else 'failed'})")
        for name, tr in ((a, ta), (bb, tb)):
            chains = tr.get(t) or []
            msgs = chains[-1][0] if chains else []
            seq = calls(msgs)
            print(f"--- {name}: {len(seq)} calls")
            for i, s in enumerate(seq, 1):
                print(f"  {i:2d}. {s}")


if __name__ == '__main__':
    main()
