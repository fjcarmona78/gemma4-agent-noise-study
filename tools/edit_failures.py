"""Count edit_file/write_file calls and their failures per recorded run, from the raw recordings in
finetune/requests/<run>.jsonl, deduplicated by tool-call id (the same response is re-sent in every later request).

  malformed: ADK rejected the call before execution ("mandatory input parameters are not present"). The usual cause
             is the model writing the arguments as backtick-delimited `key: value` text inside the JSON, so
             old_string is swallowed into filepath (2026-10-17: ~52% of our prompt's edit calls vs ~15% of the
             public bundle's, which spells out how to fill edit_file).
  no-match:  the tool ran but changed nothing (old_string not found, or not unique, or another tool error).

    python tools/edit_failures.py RUN [RUN ...]
"""
import collections
import json
import re
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1]
NOMATCH = re.compile(r'"occurrences": 0|not found in|No match|found 0|multiple matches|"status": "error"')


def load(run: str):
    calls, resp = {}, {}
    for line in (P / 'finetune' / 'requests' / f'{run}.jsonl').read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        for m in (r.get('request') or {}).get('messages') or []:
            if m.get('role') == 'assistant':
                for tc in m.get('tool_calls') or []:
                    if tc['function']['name'] in ('edit_file', 'write_file'):
                        calls[tc.get('id')] = tc['function']['arguments']
            elif m.get('role') == 'tool' and m.get('tool_call_id') in calls:
                c = m.get('content')
                if isinstance(c, list):
                    c = ' '.join(x.get('text', '') for x in c if isinstance(x, dict))
                resp[m['tool_call_id']] = c or ''
    return calls, resp


def main():
    print(f"{'run':30s} {'edit calls':>10s} {'malformed':>9s} {'no-match':>8s}   most common malformed argument keys")
    for run in sys.argv[1:]:
        calls, resp = load(run)
        mal = [i for i, c in resp.items() if 'mandatory input parameters' in c]
        nomatch = [i for i, c in resp.items() if i not in mal and NOMATCH.search(c)]
        pat = collections.Counter()
        for i in mal:
            a = calls[i]
            try:
                d = json.loads(a) if isinstance(a, str) else a
                pat[','.join(sorted(d)) if isinstance(d, dict) else 'non-dict'] += 1
            except (json.JSONDecodeError, TypeError):
                pat['unparseable'] += 1
        top = {k[:40]: v for k, v in pat.most_common(2)}
        print(f"{run:30s} {len(resp):10d} {len(mal):9d} {len(nomatch):8d}   {top}")


if __name__ == '__main__':
    main()
