"""Build fine-tuning examples from successful agent runs, token-exact to what vLLM served.

Two sources, both checked token for token against what the model actually saw:
- Recorded runs (finetune/requests/<run>.jsonl, written by finetune/proxy.py): the exact requests
  and responses, including the contexts after mid-task compaction.
- Older runs' traces (results/<run>/traces/): converted to chat messages. Each step's context must
  re-render to exactly the prompt_tokens the trace recorded; an example stops at the first step
  that does not (a compaction happened there, and traces do not contain the summary).

How the template lays out an episode (checked against vLLM's /tokenize on all 81 recorded agent
requests): one model turn; each step is `<|channel>thought\n…<channel|><|tool_call>…<tool_call|>`
followed by `<|tool_response>…<tool_response|>`; generation stops at <|tool_response> (id 50) or
<turn|> (106). Within a compaction chain every request is a token prefix of the next, so one
sequence covers all of the chain's steps; loss goes on each step's generated tokens (incl. the stop
token), never on the prompt or tool output.

    python finetune/build_dataset.py check   # probe tasks: trace-built vs recorded examples
    python finetune/build_dataset.py build [--max-tokens 12288] [--max-per-task 4] [--out PATH]

Output lines: {"task", "run", "source", "input_ids", "spans": [[start, end], ...] (loss on
input_ids[start:end]), "steps", "steps_total", "completion_tokens"}.
"""

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path.home() / 'models' / 'gemma-4-31b-it-qat-w4a16-ct'
REQUESTS = PROJECT / 'finetune' / 'requests'
PROBE_LOG = REQUESTS / 'format_probe.jsonl'

# Runs made with the deployed v3 configuration (v3 prompt, temperature 0.2, thinking budget 4096).
# Runs that changed the prompt or thinking budget are left out so the data is on-policy.
TRACE_RUNS = ['step1C_v3_dev91_t4', 'dev91_v3_t5', 'dev91_v3_t5_rep', 'dev45_v3_rerun', 'zero_lora_smoke']
# eval30 tasks are the held-out test set.
EVAL30 = set((PROJECT / 'eval30_tasks.txt').read_text().split())

# ADK appends the agent identity to the instruction; traces store the instruction without it.
SYSTEM_SUFFIX = '\n\n\nYou are an agent. Your internal name is "swe_baseline_agent".'
STOP_IDS = {1, 50, 106}  # <eos>, <|tool_response>, <turn|>
# Traces strip trailing whitespace from the thinking text; the served history kept it. Recorded
# probe steps ended with '\n\n' 35x, '\n' 18x, '' 12x, '\n\n\n' 11x, so the most common one is
# restored. Each newline run is a single token, so a trace example can differ from what was
# served only in that one token before <channel|> per step (`check` verifies exactly this).
REASONING_TAIL = '\n\n'


def parse_field(value):
    """Trace fields are JSON objects in some runs and Python reprs in others."""
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def load_tools(log: Path = PROBE_LOG) -> list[dict]:
    """Tool schemas exactly as the harness sends them (same for every submission of one family:
    PROBE_LOG for v3, V4_PROBE_LOG for v4)."""
    for line in log.read_text().splitlines():
        return json.loads(line)['request']['tools']
    raise SystemExit(f'no recorded requests in {log}')


def template_messages(messages: list[dict]) -> list[dict]:
    """Messages as vLLM 0.19.1 hands them to the chat template (entrypoints/chat_utils.py:
    _parse_chat_message_content + _postprocess_messages with the "openai" content format):
    content becomes a list of text parts, tool-call arguments become dicts, `reasoning` is copied
    to `reasoning_content`. Passing plain strings instead renders differently (e.g. a missing
    space after the system prompt), so tokens would not match what the model was served."""
    out = []
    for m in messages:
        content = m.get('content')
        if content is None:
            content = []
        elif isinstance(content, str):
            content = [{'type': 'text', 'text': content}]
        msg = {'role': m['role'],
               'content': [{'type': 'text', 'text': p['text']} for p in content if p.get('type') == 'text']}
        if m['role'] == 'assistant':
            calls = m.get('tool_calls')
            if calls:
                msg['tool_calls'] = []
                for call in calls:
                    call = json.loads(json.dumps(call))
                    args = call['function'].get('arguments')
                    call['function']['arguments'] = (args if isinstance(args, (dict, list)) else json.loads(args)) if args else {}
                    msg['tool_calls'].append(call)
            if m.get('reasoning') is not None:
                msg['reasoning'] = msg['reasoning_content'] = m['reasoning']
        elif m['role'] == 'tool' and 'tool_call_id' in m:
            msg['tool_call_id'] = m['tool_call_id']
        if isinstance(m.get('name'), str):
            msg['name'] = m['name']
        out.append(msg)
    return out


def render(tokenizer, messages: list[dict], tools: list[dict], generation_prompt: bool = False) -> str:
    return tokenizer.apply_chat_template(template_messages(messages), tools=tools, tokenize=False,
                                         add_generation_prompt=generation_prompt, enable_thinking=True)


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)['input_ids']


def final_answer_tokens(tokenizer, message: dict, completion_tokens: int | None) -> list[int] | None:
    """Tokens the model generated for a closing text answer (no tool call).

    After a tool result the prompt ends inside an opened thought channel, so the model writes
    [thinking] `<channel|>` answer `<turn|>`. The history rendering drops an empty thought
    channel, so this step is rebuilt in its generation-time form; it must match the served
    completion size exactly (it did for the probe: 80 = 80 tokens)."""
    text = (message.get('reasoning') or '') + '<channel|>' + (message.get('content') or '') + '<turn|>'
    ids = encode(tokenizer, text)
    return ids if completion_tokens is None or len(ids) == completion_tokens else None


def make_example(tokenizer, tools, messages, steps, max_tokens):
    """messages: the whole conversation incl. the last step's response; steps: one
    (context message count, prompt tokens the model saw, completion tokens) per step, in order.
    Returns (input_ids, spans, note). Stops at the first step whose context does not re-render
    to the served prompt size, or that would not fit in max_tokens."""
    full = encode(tokenizer, render(tokenizer, messages, tools))
    spans, note, ids = [], 'complete', None
    for k, (count, prompt_tokens, completion_tokens) in enumerate(steps):
        context = encode(tokenizer, render(tokenizer, messages[:count], tools, generation_prompt=True))
        if prompt_tokens is not None and len(context) != prompt_tokens:
            note = f'context {len(context)} tokens != served {prompt_tokens} (compaction)'
            break
        if full[:len(context)] != context:
            last = messages[count] if count < len(messages) else {}
            answer = (final_answer_tokens(tokenizer, last, completion_tokens)
                      if k == len(steps) - 1 and last.get('role') == 'assistant' and not last.get('tool_calls') else None)
            if answer is None:
                note = 'context is not a prefix of the episode'
            elif len(context) + len(answer) > max_tokens:
                note = f'truncated at {max_tokens} tokens'
            else:
                ids = context + answer
                spans.append([len(context), len(ids)])
            break
        end = next((i + 1 for i in range(len(context), len(full)) if full[i] in STOP_IDS), None)
        if end is None:
            note = 'no stop token after the last step'
            break
        if end > max_tokens:
            note = f'truncated at {max_tokens} tokens'
            break
        spans.append([len(context), end])
    if ids is None:
        ids = full[:spans[-1][1]] if spans else []
    return ids, spans, note


def trace_to_messages(trace: dict) -> tuple[list[dict], list[tuple]]:
    """Chat messages for a trace + one (context message count, prompt tokens, completion tokens)
    per agent step. Raises ValueError for shapes the conversion does not handle."""
    messages, steps, n = [], [], 0
    for step in trace['steps']:
        source, text = step.get('source'), step.get('message')
        if source == 'system':
            messages.append({'role': 'system', 'content': text + SYSTEM_SUFFIX})
        elif source == 'user':
            if any(m['role'] != 'system' for m in messages):
                raise ValueError('user message after the agent started')
            messages.append({'role': 'user', 'content': text})
        elif source == 'agent':
            metrics = parse_field(step.get('metrics')) or {}
            steps.append((len(messages), metrics.get('prompt_tokens'), metrics.get('completion_tokens')))
            calls = parse_field(step.get('tool_calls')) or []
            if not calls:
                messages.append({'role': 'assistant', 'content': text or ''})
                continue
            if len(calls) > 1:
                raise ValueError('step with several tool calls')
            n += 1
            msg = {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': f'call_{n}', 'type': 'function', 'function': {
                    'name': calls[0]['function_name'],
                    'arguments': json.dumps(calls[0].get('arguments') or {}, ensure_ascii=False)}}]}
            if text:
                msg['reasoning'] = text + REASONING_TAIL
            messages.append(msg)
            observation = parse_field(step.get('observation')) or {}
            messages.append({'role': 'tool', 'tool_call_id': f'call_{n}',
                             'content': served_tool_content(observation.get('content', ''))})
    return messages, steps


def served_tool_content(content: str) -> str:
    """Traces store tool results JSON-encoded with non-ASCII escaped (\\ud83d\\ude80); the model
    was sent the literal characters (🚀). Re-encoding matched all 41 probe tool results."""
    try:
        return json.dumps(json.loads(content), ensure_ascii=False)
    except (ValueError, TypeError):
        return content


AGENT_MARK = 'Your internal name is "swe_baseline_agent"'


def is_main_agent(rec: dict) -> bool:
    messages = rec['request'].get('messages') or []
    return bool(messages) and messages[0]['role'] == 'system' and any(
        m in str(messages[0]['content']) for m in (AGENT_MARK, 'Your internal name is "swe_agent"'))  # public bundle's agent name


def split_tasks(records: list[dict]) -> list[list[dict]]:
    """Group proxy records into tasks (the harness runs tasks one at a time; each starts with
    [system, user]). Requests of the code_analyzer sub-agent (its own system prompt) are dropped;
    summarization requests (no tools) are kept as compaction markers."""
    tasks = []
    # The proxy logs a request when its response completes. A request still generating when a
    # task hits its time limit finishes (up to ~80 s later) inside the next task's log, so
    # restore the order requests were sent in.
    for rec in sorted(records, key=lambda r: r['ts']):
        if rec['request'].get('tools') and not is_main_agent(rec):
            continue
        if is_main_agent(rec) and len(rec['request']['messages']) == 2:
            tasks.append([])
        if tasks:
            tasks[-1].append(rec)
    return tasks


def split_chains(task: list[dict]) -> tuple[list[list[dict]], list[dict]]:
    """Split a task's agent requests into chains separated by context compaction.

    ADK compacts mid-task once the prompt passes ~14k tokens: one summarization request (single
    user message, no tools), then the context restarts as system + "For context: [model] said:
    <summary>" + the last few steps. The original problem statement is gone after that.
    """
    chains, summaries = [[]], []
    for rec in task:
        if not rec['request'].get('tools'):
            summaries.append(rec)
            chains.append([])
        else:
            chains[-1].append(rec)
    return [c for c in chains if c], summaries


def chain_to_messages(chain: list[dict]) -> tuple[list[dict], list[tuple]]:
    """The chain's last request + its response (plus a placeholder tool result so the stop token
    renders), and one (context message count, prompt tokens, completion tokens) per request."""
    last = chain[-1]
    response = last['response']['choices'][0]['message']
    messages = list(last['request']['messages'])
    msg = {'role': 'assistant', 'content': response.get('content')}
    if response.get('tool_calls'):
        msg['tool_calls'] = response['tool_calls']
    if response.get('reasoning'):
        msg['reasoning'] = response['reasoning']
    messages.append(msg)
    for call in response.get('tool_calls') or []:
        messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': ''})
    steps = [(len(r['request']['messages']), r['response']['usage']['prompt_tokens'],
              r['response']['usage']['completion_tokens']) for r in chain]
    for rec in chain:
        if messages[:len(rec['request']['messages'])] != rec['request']['messages']:
            raise ValueError('chain is not append-only')
    return messages, steps


def run_results(run: str) -> dict[str, dict]:
    path = PROJECT / 'results' / run / 'task_results.jsonl'
    return {r['instance_id']: r for r in map(json.loads, path.read_text().splitlines()) if r}


def load_trace(run: str, task: str) -> dict | None:
    path = PROJECT / 'results' / run / 'traces' / f'trace_{task}.json'
    return json.loads(path.read_text()) if path.exists() else None


def recorded_trajectories(run: str, include_failed: bool = False):
    """(task, [(messages, steps)] per chain) for each resolved dev task of a recorded run (with
    include_failed, for every dev task: KTO needs the failed trajectories too)."""
    results = run_results(run)
    by_user = {}
    for task, row in results.items():
        trace = load_trace(run, task)
        if trace and trace.get('steps'):
            by_user[next(s['message'] for s in trace['steps'] if s.get('source') == 'user')] = task
    records = []
    for line in (REQUESTS / f'{run}.jsonl').read_text().splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:  # only a log still being written can end in a partial line
            pass
    for task_records in split_tasks(records):
        task = by_user.get(task_records[0]['request']['messages'][1]['content'])
        if task is None or task in EVAL30 or not (include_failed or results[task]['resolved']):
            continue
        if any(r['status'] != 200 or not isinstance(r['response'], dict) or not r['response'].get('choices')
               for r in task_records):
            print(f'{run}/{task}: failed requests (e.g. vLLM restart); skipped')
            continue
        chains, _ = split_chains(task_records)
        converted = []
        for k, chain in enumerate(chains):
            try:
                converted.append(chain_to_messages(chain))
            except ValueError as error:
                print(f'{run}/{task} chain {k}: {error}; skipped')
        yield task, converted


def trace_trajectories(run: str):
    for task, row in run_results(run).items():
        if task in EVAL30 or not row['resolved']:
            continue
        trace = load_trace(run, task)
        if trace and trace.get('steps'):
            yield task, [trace_to_messages(trace)]


def served_config(log: Path) -> tuple[str, str] | None:
    """(system prompt, model name) of a recorded run's first main-agent request."""
    for line in log.read_text().splitlines():
        try:
            rec = json.loads(line)
        except ValueError:  # a partial last line while the run is still being recorded
            continue
        if is_main_agent(rec):
            return rec['request']['messages'][0]['content'], rec['request'].get('model')
    return None


PINNED_BLOCK = '\n## The task statement'
ADK_SUFFIX = '\n\n\nYou are an agent.'  # ADK appends its own agent line after the submission's prompt


def prompt_key(system: str) -> str:
    """The served system prompt without the per-task pinned statement block, so the pinned and unpinned
    variants of one prompt compare equal (submissions/v4c_pinned vs v4c)."""
    i = system.find(PINNED_BLOCK)
    if i < 0:
        return system
    j = system.find(ADK_SUFFIX, i)
    return system[:i].rstrip() + (system[j:] if j >= 0 else '')  # ADK strips the prompt's trailing newline


# Recorded runs of the exact deployed configuration (frozen v3 prompt + sampling, base 31B). An
# explicit list: some experiments kept the v3 prompt but changed the request format or sampling
# (dev91_dropthoughts drops history reasoning; dev91_gemmasampling ran at temperature 1.0).
RECORDED_RUNS = ['format_probe', 'dev91_v3_t4_rec', 'dev91_v3_rec2', 'dev91_v3_rec3', 'dev91_hinted1']
# Route C: runs on data/tasks_hinted.jsonl, whose statements end with a hint saying WHERE the fix goes
# (runner/make_hints.py). Their successes are used with the hint removed (strip_hints).
HINTED_RUNS = {'dev91_hinted1'}
# Profile "v4" (adapter v3): data in the format of the v4-family submission now deployed, v4b
# (submissions/v4b: v4's single agent and prompt on v3's budget). Gemma's own runs of exactly that
# submission, token-verified like the runs above, plus local TEACHER runs of the same submission with
# only the model (and its recommended sampling) swapped: experiments/v4b_teacher_qwen36, Qwen3.6-27B
# (Apache-2.0; the host allows other open models' trajectories). A teacher's served prompt/completion
# sizes come from its own tokenizer, so its examples are rendered with Gemma's template without the
# served-size check (the prefix and stop-token checks still apply).
V4_PROBE_LOG = REQUESTS / 'dev91_v4b.jsonl'
V4_RECORDED_RUNS = ['dev91_v4b']
V4_FAMILY_LOGS = [REQUESTS / 'dev91_v4c.jsonl']  # other prompts of the v4 family a teacher run may have served
TEACHER_RUNS = ['dev91_teacher_qwen36', 'v2_teacher_qwen36']  # a listed run is skipped until it exists
HINT_RE = re.compile(re.escape('\n\nHint (from the task authors): ') + r'the complete fix changes .*?Make all of these changes\.', re.S)


# The model's own mentions of the hint ("The hint says the fix is in ..."). "type hints" is code
# vocabulary, not the hint; the harness prompt's "any provided hints" is in every task's user message
# and is never scrubbed (only assistant messages and compaction summaries are).
HINT_WORD = re.compile(r'(?<!type )(?<!type-)\bhint(?:s|ed)?\b|task authors', re.I)


def mentions_hint(message: dict) -> bool:
    return bool(HINT_WORD.search(json.dumps([message.get('reasoning'), message.get('content')])))


def scrub_text(text: str) -> str:
    """`text` without the sentences that mention the hint (other lines kept as they are)."""
    lines = []
    for line in text.split('\n'):
        if HINT_WORD.search(line):
            line = ' '.join(s for s in re.split(r'(?<=[.!?:])\s+', line) if not HINT_WORD.search(s))
            if not line.strip():
                continue
        lines.append(line)
    return '\n'.join(lines)


def scrub_message(message: dict) -> dict:
    out = dict(message)
    for key in ('reasoning', 'content'):
        value = out.get(key)
        if isinstance(value, str):
            out[key] = scrub_text(value)
        elif isinstance(value, list):
            out[key] = [{**p, 'text': scrub_text(p['text'])} if isinstance(p, dict) and isinstance(p.get('text'), str) else p
                        for p in value]
    return out


def strip_hints(chains, tokenizer, tools):
    """Remove the Route C hint from a hinted run so the example shows the model working from the
    original statement. The hint block is cut from the problem statement and the model's mentions
    of it are cut from its reasoning and compaction summaries; the steps that mentioned it stay in
    the context but get no loss, so the model is never trained to make a hint-driven jump (e.g. 'Let's
    start by examining rich/cells.py' right after the statement), only on the steps that follow.
    Each step's expected prompt size is shifted by the exact token difference of its edited context.
    (The model names the hint in its very first reasoning, so cutting each example at the first
    mention, as adapter v2's first draft did, kept nothing: 0 of 2 successes on 2026-10-13.)"""
    if not chains or not isinstance(chains[0][0][1].get('content'), str) or not HINT_RE.search(chains[0][0][1]['content']):
        return []
    out = []
    for k, (messages, steps) in enumerate(chains):
        new = list(messages)
        new[1] = ({**new[1], 'content': HINT_RE.sub('', new[1]['content'], count=1)} if k == 0
                  else scrub_message(new[1]))
        masked = {c for c, _, _ in steps if c < len(messages) and mentions_hint(messages[c])}
        new = [scrub_message(m) if m.get('role') == 'assistant' and mentions_hint(m) else m for m in new]
        kept = []
        for c, p, ct in steps:
            if c in masked:
                continue
            if p is not None:
                p -= (len(encode(tokenizer, render(tokenizer, messages[:c], tools, True)))
                      - len(encode(tokenizer, render(tokenizer, new[:c], tools, True))))
            kept.append((c, p, ct))
        if kept:
            out.append((new, kept))
    return out


def malformed_call(message: dict, schema: dict) -> bool:
    """A tool call whose arguments do not fit the tool's schema: the model broke the call syntax
    (e.g. closed a string with a backtick) and vLLM's parser turned code into argument names. The
    harness answers with an error and the model retries; the retry is trained, the broken call stays
    in the context with no loss (25 of 1140 steps of adapter v2's runs, 10 in one rich_4075 run)."""
    for call in message.get('tool_calls') or []:
        name, args = call['function']['name'], call['function'].get('arguments')
        try:
            args = args if isinstance(args, dict) else json.loads(args or '{}')
        except ValueError:
            return True
        if name not in schema or not isinstance(args, dict):
            return True
        properties, required = schema[name]
        if set(args) - properties or required - set(args):
            return True
    return False


def recorded_run(run: str) -> bool:
    return (REQUESTS / f'{run}.jsonl').exists() and (PROJECT / 'results' / run / 'task_results.jsonl').exists()


def fold_visible_text(messages: list[dict]) -> list[dict]:
    """Teacher turns (Qwen) often carry visible text next to their tool calls; Gemma's turns never do.
    Gemma's chat template renders such text after the tool responses as the END of the model turn, so
    the next step no longer continues the turn and the builder would drop every later step. The text
    is moved into the reasoning channel (after the thinking, as the teacher produced it): the
    Gemma-native place for a diagnosis stated before a tool call."""
    out = []
    for m in messages:
        text = m.get('content') if isinstance(m.get('content'), str) else None
        if m.get('role') == 'assistant' and m.get('tool_calls') and text and text.strip():
            m = dict(m)
            reasoning = (m.get('reasoning') or m.get('reasoning_content') or '').rstrip()
            m['reasoning'] = m['reasoning_content'] = (reasoning + '\n\n' + text.strip()) if reasoning else text.strip()
            m['content'] = None
        out.append(m)
    return out


# Profile "e4b" (2026-10-24, user-approved): distil the 31B into Gemma 4 E4B. Teacher = Gemma 31B's own
# SOLVED, recorded runs of the two 0.18-family archives (public no-edit bundle, safe prompt), each with the
# system prompt and tools it served. Held out, by task AND by repository: the E4B evaluation set (recent-33),
# the protected final set, the OOD-45 overlap repos; rebench-dev is the test set's development split, so
# never training data (assert_no_bench_repos also enforces that).
E4B_RUN_RE = re.compile(r'^(v2t74|fresh15|ood45)_(public_bav2_noedit_t7|v4c_pinned_cap28_noedit_safe)(_rep\d*)?$')
TEACHER_MODEL = 'gemma-4-31b-it-qat-w4a16-ct'
OOD45_OVERLAP_REPOS = {'pint', 'tox', 'pdm', 'pyjanitor', 'nox'}  # repo names (owner varies); never training data


def e4b_heldout():
    ids, repos = set(), set()
    for tasks, id_file in [('data/rebench_recent/tasks.jsonl', 'data/rebench_recent/eval_ids.txt'),
                           ('data/rebench_scale/tasks.jsonl', 'data/rebench_scale/final_ids.txt')]:
        keep = set((PROJECT / id_file).read_text().split())
        for line in (PROJECT / tasks).read_text().splitlines():
            t = json.loads(line)
            if t['instance_id'] in keep:
                ids.add(t['instance_id'])
                repos.add(t['repo'].lower())
    if BENCH_REPOS_FILE.exists():
        repos |= {r.lower() for r in json.loads(BENCH_REPOS_FILE.read_text())}
    return ids, repos


def e4b_examples(tokenizer, max_tokens):
    held_ids, held_repos = e4b_heldout()
    runs = sorted(p.stem for p in REQUESTS.glob('*.jsonl') if E4B_RUN_RE.match(p.stem) and recorded_run(p.stem))
    for run in runs:
        log = REQUESTS / f'{run}.jsonl'
        served = served_config(log)
        if not served or served[1] != TEACHER_MODEL:
            print(f'{run}: not served by the 31B; skipped')
            continue
        tools = load_tools(log)
        schema = {t['function']['name']: (set(t['function']['parameters'].get('properties', {})),
                                          set(t['function']['parameters'].get('required', []))) for t in tools}
        repo_of = {t: (row.get('repo') or '').lower() for t, row in run_results(run).items()}
        for task, chains in recorded_trajectories(run):
            repo = repo_of.get(task, '')
            if task in held_ids or repo in held_repos or repo.split('/')[-1] in OOD45_OVERLAP_REPOS:
                continue
            for k, (messages, steps) in enumerate(chains):
                trained = [s for s in steps if not malformed_call(messages[s[0]] if s[0] < len(messages) else {}, schema)]
                ids, spans, note = make_example(tokenizer, tools, messages, trained, max_tokens)
                yield {'task': task, 'run': run, 'source': 'recorded', 'chain': k, 'input_ids': ids, 'spans': spans,
                       'steps': len(spans), 'steps_total': len(steps), 'note': note,
                       'completion_tokens': sum(s[2] or 0 for s in steps)}


def build_examples(tokenizer, tools, max_tokens, profile='v3'):
    if profile == 'e4b':
        yield from e4b_examples(tokenizer, max_tokens)
        return
    if profile == 'v4':
        # Gemma's runs must have served exactly v4's prompt with the base 31B; teacher runs the same
        # prompt (the model name differs by design).
        v4 = served_config(V4_PROBE_LOG)
        sources = [(run, 'recorded', recorded_trajectories) for run in V4_RECORDED_RUNS
                   if recorded_run(run) and served_config(REQUESTS / f'{run}.jsonl') == v4]
        # Teacher runs may use any prompt of the v4 family (v4b, v4c, or their pinned variants): the
        # adapter learns the trajectories, and the deployed prompt is one of these.
        family = {prompt_key(v4[0])} | {prompt_key(served_config(log)[0]) for log in V4_FAMILY_LOGS if log.exists()}
        sources += [(run, 'teacher', recorded_trajectories) for run in TEACHER_RUNS
                    if recorded_run(run) and prompt_key((served_config(REQUESTS / f'{run}.jsonl') or ('',))[0]) in family]
    else:
        # Each listed run must also have served the probe's prompt with the base 31B.
        v3 = served_config(PROBE_LOG)
        recorded = [run for run in RECORDED_RUNS
                    if recorded_run(run) and served_config(REQUESTS / f'{run}.jsonl') == v3]
        sources = [(run, 'recorded', recorded_trajectories) for run in recorded]
        sources += [(run, 'trace', trace_trajectories) for run in TRACE_RUNS if run not in recorded]
    schema = {t['function']['name']: (set(t['function']['parameters'].get('properties', {})),
                                      set(t['function']['parameters'].get('required', []))) for t in tools}
    for run, kind, fn in sources:
        for task, chains in fn(run):
            if run in HINTED_RUNS:
                chains = strip_hints(chains, tokenizer, tools)
            for k, (messages, steps) in enumerate(chains):
                trained = [s for s in steps if not malformed_call(messages[s[0]] if s[0] < len(messages) else {}, schema)]
                if kind == 'teacher':  # sizes from the teacher's tokenizer: nothing to verify against
                    trained = [(c, None, None) for c, _, _ in trained]
                    messages = fold_visible_text(messages)
                ids, spans, note = make_example(tokenizer, tools, messages, trained, max_tokens)
                yield {'task': task, 'run': run, 'source': kind, 'chain': k, 'input_ids': ids, 'spans': spans,
                       'steps': len(spans), 'steps_total': len(steps), 'note': note,
                       'completion_tokens': sum(s[2] or 0 for s in steps)}


# Hard rule (user, 2026-10-13): no training example may come from a repository of the rebench test set
# (development or protected split). Public run datasets overlap it heavily (Nebius: 69 of the 78 test tasks).
BENCH_REPOS_FILE = PROJECT / 'data' / 'rebench_scale' / 'bench_repos.json'


def assert_no_bench_repos(examples) -> None:
    if not BENCH_REPOS_FILE.exists():
        return
    bench = set(json.loads(BENCH_REPOS_FILE.read_text()))
    repos = {run: {t: row.get('repo') for t, row in run_results(run).items()} for run in {e['run'] for e in examples}}
    bad = sorted({(e['run'], e['task']) for e in examples if repos[e['run']].get(e['task']) in bench})
    if bad:
        raise SystemExit(f'refusing to build: {len(bad)} examples come from test-set repositories, e.g. {bad[:3]}')


# KTO (Ethayarajh et al. 2024) on Gemma's OWN trajectories: every chain of every dev91 task in these
# recorded runs, labelled by whether the run solved the task. Only runs of the v4 prompt family
# (single agent, 6 tools), the family of the shipping prompt; rebench and recent runs are evaluation
# sets and never enter training.
KTO_RUNS = ['dev91_v4b', 'dev91_v4c']


def build_kto(args) -> None:
    """Writes examples with a `label` (True = the run solved the task). Desirable examples are
    supervised like SFT (malformed tool calls masked); undesirable ones keep every generated step,
    malformed calls included, since those are part of what the label disapproves of."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    v4 = served_config(V4_PROBE_LOG)
    family = {prompt_key(v4[0])} | {prompt_key(served_config(log)[0]) for log in V4_FAMILY_LOGS if log.exists()}
    kept, skipped = [], Counter()
    for run in (args.runs or KTO_RUNS):
        log = REQUESTS / f'{run}.jsonl'
        served = served_config(log) if recorded_run(run) else None
        if not served or prompt_key(served[0]) not in family or served[1] != v4[1]:
            print(f'{run}: not a recorded v4-family run of the base model; skipped')
            continue
        tools = load_tools(log)
        schema = {t['function']['name']: (set(t['function']['parameters'].get('properties', {})),
                                          set(t['function']['parameters'].get('required', []))) for t in tools}
        resolved = {t: r['resolved'] for t, r in run_results(run).items()}
        for task, chains in recorded_trajectories(run, include_failed=True):
            label = bool(resolved[task])
            for k, (messages, steps) in enumerate(chains):
                trained = [s for s in steps if not (label and malformed_call(messages[s[0]] if s[0] < len(messages) else {}, schema))]
                ids, spans, note = make_example(tokenizer, tools, messages, trained, args.max_tokens)
                if not spans:
                    skipped[note.split(' (')[0]] += 1
                    continue
                kept.append({'task': task, 'run': run, 'source': 'recorded', 'chain': k, 'label': label,
                             'input_ids': ids, 'spans': spans, 'steps': len(spans), 'steps_total': len(steps),
                             'note': note, 'completion_tokens': sum(s[2] or 0 for s in steps)})
    assert_no_bench_repos(kept)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        for e in kept:
            f.write(json.dumps(e) + '\n')
    outcomes = defaultdict(set)
    for e in kept:
        outcomes[e['task']].add(e['label'])
    report = {
        'examples': len(kept), 'desirable': sum(e['label'] for e in kept), 'undesirable': sum(not e['label'] for e in kept),
        'tasks': len(outcomes), 'tasks_with_both_labels': sum(len(v) == 2 for v in outcomes.values()),
        'tokens': sum(len(e['input_ids']) for e in kept), 'loss_tokens': sum(b - a for e in kept for a, b in e['spans']),
        'loss_tokens_desirable': sum(b - a for e in kept if e['label'] for a, b in e['spans']),
        'steps_used': sum(e['steps'] for e in kept), 'steps_total': sum(e['steps_total'] for e in kept),
        'stop_reasons': Counter(e['note'].split(' (')[0] if 'compaction' not in e['note'] else 'compaction' for e in kept),
        'skipped_no_spans': skipped, 'per_run': Counter(e['run'] for e in kept), 'max_tokens': args.max_tokens,
    }
    out.with_suffix('.report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def build(args) -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    tools = None if args.profile == 'e4b' else load_tools(V4_PROBE_LOG if args.profile == 'v4' else PROBE_LOG)
    examples = [e for e in build_examples(tokenizer, tools, args.max_tokens, args.profile)]
    assert_no_bench_repos(examples)
    # Keep at most N trajectories per task, the most efficient ones (fewest generated tokens),
    # so tasks every run solves do not dominate and the model sees the quickest solutions.
    # Adapter v2 uses N=1: only the fastest successful run per task. A task solved both with and
    # without a hint (e.g. rich_3506/rich_3676, solved by v3 repeat 71) uses the unhinted runs first,
    # and in the v4 profile a teacher run comes before Gemma's own (the teacher is the new information).
    per_task = defaultdict(lambda: defaultdict(list))
    for e in examples:
        per_task[e['task']][(e['run'], e['source'])].append(e)
    kept = []
    for task, runs in per_task.items():
        ranked = sorted(runs.values(), key=lambda chains: (chains[0]['run'] in HINTED_RUNS,
                                                           chains[0]['source'] != 'teacher',
                                                           sum(c['completion_tokens'] for c in chains)))
        for chains in ranked[:args.max_per_task]:
            kept.extend(c for c in chains if c['spans'])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        for e in kept:
            f.write(json.dumps(e) + '\n')
    lengths = sorted(len(e['input_ids']) for e in kept)
    loss_tokens = sum(b - a for e in kept for a, b in e['spans'])
    report = {
        'examples': len(kept), 'trajectories_found': sum(len(r) for r in per_task.values()),
        'tasks': len(per_task), 'tokens': sum(lengths), 'loss_tokens': loss_tokens,
        'length_median': lengths[len(lengths) // 2] if lengths else 0, 'length_max': lengths[-1] if lengths else 0,
        'steps_used': sum(e['steps'] for e in kept), 'steps_total': sum(e['steps_total'] for e in kept),
        'stop_reasons': Counter(e['note'].split(' (')[0] if 'compaction' not in e['note'] else 'compaction' for e in kept),
        'per_run': Counter(e['run'] for e in kept), 'per_source': Counter(e['source'] for e in kept),
        'max_tokens': args.max_tokens, 'max_per_task': args.max_per_task,
    }
    out.with_suffix('.report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def check() -> None:
    """For the probe tasks, the example built from the trace must equal the recorded one."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    tools = load_tools()
    recorded = {task: chains for task, chains in recorded_trajectories('format_probe')}
    for task, chains in trace_trajectories('format_probe'):
        messages, steps = chains[0]
        t_ids, t_spans, t_note = make_example(tokenizer, tools, messages, steps, 10 ** 6)
        r_messages, r_steps = recorded[task][0]
        r_ids, r_spans, r_note = make_example(tokenizer, tools, r_messages, r_steps, 10 ** 6)
        n = len(t_spans)
        if t_spans != r_spans[:n] or len(t_ids) > len(r_ids):
            print(f'{task}: DIFFERENT step layout ({n} vs {len(r_spans)} steps)')
            continue
        diffs = [i for i, (a, b) in enumerate(zip(t_ids, r_ids)) if a != b]
        channel_end = tokenizer.convert_tokens_to_ids('<channel|>')
        whitespace = [i for i in diffs if not tokenizer.decode([t_ids[i]]).strip() and not tokenizer.decode([r_ids[i]]).strip()
                      and t_ids[i + 1] == r_ids[i + 1] == channel_end]
        print(f'{task}: {n} steps ({t_note}; recorded {len(r_spans)} steps, {r_note}); {len(t_ids)} tokens, '
              f'{len(diffs)} differ, of which {len(whitespace)} are the newline run before <channel|>'
              f'{"" if len(diffs) == len(whitespace) else "  <-- OTHER DIFFERENCES"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['check', 'build', 'build-kto'])
    parser.add_argument('--max-tokens', type=int, default=12288)
    parser.add_argument('--max-per-task', type=int, default=4)
    parser.add_argument('--profile', choices=['v3', 'v4', 'e4b'], default='v3',
                        help='v3: frozen-v3 runs and traces (adapters v1/v2); v4: v4 runs + teacher runs (adapter v3)')
    parser.add_argument('--out', default=str(PROJECT / 'finetune' / 'data' / 'train.jsonl'))
    parser.add_argument('--runs', nargs='+', help='build-kto only: recorded runs to label (default: KTO_RUNS)')
    args = parser.parse_args()
    if args.command == 'check':
        check()
    elif args.command == 'build-kto':
        build_kto(args)
    else:
        build(args)
