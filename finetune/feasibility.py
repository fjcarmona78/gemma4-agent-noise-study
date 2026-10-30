"""Fine-tuning feasibility test for the RTX 5090 (run in the `ft` env while vLLM is stopped).

Answers: does LoRA training of the competition model fit in 32 GB, how fast is it, and does the
saved adapter use the tensor names the scorer's vLLM expects?

    python finetune/feasibility.py            # parent: runs load strategies until one succeeds
    python finetune/feasibility.py --dry-run  # CPU only: checks data building + adapter renaming

Each load strategy runs in its own subprocess so a failed attempt cannot leave GPU memory
fragmented. Results accumulate in finetune/feasibility/report.json.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / 'finetune' / 'feasibility'
# Optional overrides for follow-up runs (inherited by the per-strategy subprocesses):
#   FEAS_TAG=_run3 keeps results separate; FEAS_SEQ_LENS=10240,12288; FEAS_ONLY=<strategy name>.
TAG = os.environ.get('FEAS_TAG', '')
REPORT = OUT / f'report{TAG}.json'
COMPETITION_MODEL = Path.home() / 'models' / 'gemma-4-31b-it-qat-w4a16-ct'
# Re-sharded copy (finetune/reshard.py): WSL cannot mmap the original 49.8 GB shard.
UNQUANTIZED_MODEL = Path.home() / 'models' / 'gemma-4-31b-it-qat-q4_0-unquantized-5gb'
COMPETITION_NAME = 'google/gemma-4-31b-it-qat-w4a16-ct'
SAMPLE_ADAPTER_PREFIX = 'base_model.model.model.language_model.'

SEQ_LENS = [int(n) for n in os.environ.get('FEAS_SEQ_LENS', '8192,16384').split(',')]
STEPS_PER_LEN = 3
LORA_RANK = 16

# (name, model path, load_in_4bit, extra environment). Tried in order until one trains at
# every sequence length. Run 1 (report_run1.json) showed: load_in_4bit=False on the competition
# checkpoint falls back to 16-bit LoRA and runs out of memory; unsloth's packed-INT4 Triton
# kernel fails to compile (float32 bit-shift); the 49.8 GB unquantized shard cannot be mmapped.
STRATEGIES = [
    ('competition_nf4_requant', COMPETITION_MODEL, True, {'UNSLOTH_COMPRESSED_TENSORS_INT4': 'nf4'}),
    ('unquantized_qat_nf4_resharded', UNQUANTIZED_MODEL, True, {}),
]

# Runs that were scored in a working sandbox; eval30 tasks are the held-out test set.
RESULT_DIRS = ['step1C_v3_dev91_t4', 'dev45_v3_rerun', 'dev45_think2048', 'dev45_think512',
               'dev45_timeaware', 'dev91_v3_t5', 'eval30_v2', 'eval30_v2_part2', 'step1B_v3_eval30']


def load_report() -> dict:
    return json.loads(REPORT.read_text()) if REPORT.exists() else {'attempts': {}}


def save_report(report: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1))


def trajectory_text() -> str:
    """Concatenated text of successful dev-task trajectories (eval30 excluded)."""
    eval30 = set((PROJECT / 'eval30_tasks.txt').read_text().split())
    chunks = []
    for d in RESULT_DIRS:
        path = PROJECT / 'results' / d / 'task_results.jsonl'
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if not row['resolved'] or row['instance_id'] in eval30:
                continue
            trace = json.loads((PROJECT / 'results' / d / 'traces' / f"trace_{row['instance_id']}.json").read_text())
            for step in trace['steps']:
                msg = step.get('message')
                chunks.append(msg if isinstance(msg, str) else json.dumps(msg))
                for call in step.get('tool_calls') or []:
                    chunks.append(json.dumps(call.get('arguments')))
                obs = step.get('observation')
                if obs:
                    chunks.append(json.dumps(obs))
    return '\n'.join(chunks)


def scorer_key(key: str) -> str:
    """Rename a PEFT tensor key to the layout of the hosts' sample adapter."""
    match = re.search(r'(layers\.\d+\..*)$', key)
    if not match:
        raise ValueError(f'unexpected adapter tensor (not a language layer): {key}')
    suffix = match.group(1).replace('.linear.lora_', '.lora_')
    return SAMPLE_ADAPTER_PREFIX + suffix


def export_for_vllm(src: Path, dst: Path, base_name: str = COMPETITION_NAME) -> dict:
    """Copy a saved adapter with scorer-style tensor names and base model name."""
    from safetensors.torch import load_file, save_file

    tensors = load_file(str(src / 'adapter_model.safetensors'))
    renamed = {scorer_key(k): v.to(dtype=v.dtype).contiguous() for k, v in tensors.items()}
    dst.mkdir(parents=True, exist_ok=True)
    save_file(renamed, str(dst / 'adapter_model.safetensors'))
    config = json.loads((src / 'adapter_config.json').read_text())
    config['base_model_name_or_path'] = base_name
    # Unsloth writes target_modules as a regex string; the hosts' sample adapter uses a plain list.
    config['target_modules'] = sorted({k.split('.')[-3] for k in renamed if '.lora_' in k})
    (dst / 'adapter_config.json').write_text(json.dumps(config, indent=1))
    size_mb = (dst / 'adapter_model.safetensors').stat().st_size / 1e6
    return {'tensors': len(renamed), 'size_mb': round(size_mb, 1), 'example_keys': sorted(renamed)[:2],
            'target_modules': config.get('target_modules'), 'r': config.get('r')}


def run_attempt(name: str, model_path: Path, load_in_4bit: bool) -> dict:
    """Load, attach LoRA, train a few steps per sequence length, save the adapter."""
    import unsloth  # noqa: F401  (must be imported before transformers)
    import torch
    from unsloth import FastModel

    result = {'model': str(model_path), 'load_in_4bit': load_in_4bit, 'steps': []}
    t0 = time.time()
    model, tokenizer = FastModel.from_pretrained(
        model_name=str(model_path), max_seq_length=max(SEQ_LENS), dtype=torch.bfloat16,
        load_in_4bit=load_in_4bit, text_only=True, use_gradient_checkpointing='unsloth')
    result['load_seconds'] = round(time.time() - t0)
    result['memory_after_load_gb'] = round(torch.cuda.memory_allocated() / 1e9, 2)

    model = FastModel.get_peft_model(
        model, r=LORA_RANK, lora_alpha=2 * LORA_RANK, lora_dropout=0.0,
        finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        use_gradient_checkpointing='unsloth', random_state=0)
    trainable = [p for p in model.parameters() if p.requires_grad]
    result['trainable_params_m'] = round(sum(p.numel() for p in trainable) / 1e6, 1)
    optimizer = torch.optim.AdamW(trainable, lr=1e-5)

    text_tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)
    ids = text_tokenizer(trajectory_text(), add_special_tokens=False)['input_ids']
    result['corpus_tokens'] = len(ids)

    model.train()
    offset = 0
    for seq_len in SEQ_LENS:
        for step in range(STEPS_PER_LEN):
            if offset + seq_len > len(ids):
                offset = 0
            batch = torch.tensor([ids[offset:offset + seq_len]], device='cuda')
            offset += seq_len
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t = time.time()
            try:
                loss = model(input_ids=batch, labels=batch).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                result['steps'].append({'seq_len': seq_len, 'step': step, 'oom': True,
                                        'peak_reserved_gb': round(torch.cuda.max_memory_reserved() / 1e9, 2)})
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                break
            seconds = time.time() - t
            result['steps'].append({
                'seq_len': seq_len, 'step': step, 'seconds': round(seconds, 1),
                'tokens_per_second': round(seq_len / seconds), 'loss': round(loss.item(), 3),
                'peak_allocated_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2),
                'peak_reserved_gb': round(torch.cuda.max_memory_reserved() / 1e9, 2)})
            print(json.dumps(result['steps'][-1]), flush=True)

    saved = OUT / f'adapter_{name}{TAG}'
    model.save_pretrained(str(saved))
    result['adapter_raw'] = str(saved)
    result['adapter_vllm'] = str(OUT / f'adapter_vllm{TAG}')
    result['adapter_export'] = export_for_vllm(saved, OUT / f'adapter_vllm{TAG}')
    return result


def child(index: int) -> None:
    name, path, load_in_4bit, env = STRATEGIES[index]
    os.environ.update(env)
    report = load_report()
    try:
        result = run_attempt(name, path, load_in_4bit)
        lengths_ok = {s['seq_len'] for s in result['steps'] if not s.get('oom')}
        result['ok'] = set(SEQ_LENS) <= lengths_ok
    except Exception:
        result = {'ok': False, 'model': str(path), 'load_in_4bit': load_in_4bit, 'env': env,
                  'error': traceback.format_exc()[-3000:]}
    report = load_report()
    report['attempts'][name] = result
    save_report(report)
    print(json.dumps({k: v for k, v in result.items() if k != 'steps'}, indent=1)[:4000], flush=True)


def parent() -> None:
    save_report({'started': time.strftime('%F %T'), 'attempts': {}})
    only = os.environ.get('FEAS_ONLY')
    for index, (name, path, _, _) in enumerate(STRATEGIES):
        if only and name != only:
            continue
        if not (path / 'config.json').exists():
            report = load_report()
            report['attempts'][name] = {'ok': False, 'error': f'model not found at {path}'}
            save_report(report)
            continue
        print(f'=== attempt {name}', flush=True)
        subprocess.run([sys.executable, __file__, '--child', str(index)], check=False)
        if load_report()['attempts'].get(name, {}).get('ok'):
            break
    report = load_report()
    report['finished'] = time.strftime('%F %T')
    report['success'] = next((n for n, a in report['attempts'].items() if a.get('ok')), None)
    save_report(report)
    print(json.dumps({'success': report['success']}), flush=True)


def dry_run() -> None:
    """CPU-only checks: corpus building, tokenizer, and adapter renaming on the sample adapter."""
    from transformers import AutoTokenizer

    text = trajectory_text()
    tokenizer = AutoTokenizer.from_pretrained(str(COMPETITION_MODEL))
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    print(f'corpus: {len(text) / 1e6:.1f} M chars, {len(ids) / 1e6:.2f} M tokens '
          f'({len(ids) // max(SEQ_LENS)} sequences of {max(SEQ_LENS)})')
    sample = PROJECT / 'data' / 'sample_submission' / 'adapters' / 'main_lora'
    tmp = OUT / 'dry_run_sample_export'
    print('sample adapter round-trip:', export_for_vllm(sample, tmp))
    for key in ['base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight',
                'base_model.model.model.language_model.layers.3.mlp.down_proj.lora_B.weight',
                'base_model.model.language_model.model.layers.3.self_attn.o_proj.linear.lora_A.weight']:
        print(f'  {key}\n    -> {scorer_key(key)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.child is not None:
        child(args.child)
    else:
        parent()
