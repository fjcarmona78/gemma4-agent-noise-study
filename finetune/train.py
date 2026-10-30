"""QLoRA fine-tuning on agent trajectories built by finetune/build_dataset.py (run in the `ft` env
while vLLM is stopped).

Base: the unquantized QAT checkpoint, re-sharded (finetune/reshard.py), loaded in 4-bit by unsloth;
the competition's compressed INT4 checkpoint cannot be trained in this env (feasibility runs 1-2).
LoRA on all 7 linear projections of the language layers; loss only on the tokens the model
generated (the dataset's spans), token-weighted across each accumulation group.

    python finetune/train.py --data finetune/data/train.jsonl --out finetune/runs/<name>
    python finetune/train.py --data ... --out ... --probe 12288,14336,16384 [--offload]

--kto trains with KTO (Ethayarajh et al. 2024) on a `build_dataset.py build-kto` dataset, whose
examples carry `label` (the run solved the task). The adapter starts at zero, so the model before the
first step IS the reference policy: one no-grad pass caches each example's reference loss, and no
second model is held in memory. The implied reward is length-normalized: r = (reference mean NLL) -
(policy mean NLL) over the example's supervised tokens. Desirable examples are pushed up with weight
lambda_D, undesirable ones down with lambda_U (default: equal total weight per class), through
1 - sigmoid(beta * (+/-(r - z))), where z is a running, non-negative estimate of the mean reward
(the paper's KL reference point). --sft-weight adds that weight times the mean NLL of each desirable
example (the NLL term of iterative RPO, Pang et al. 2024): in KTO v1 the desirable chains' likelihood
fell along with the undesirable ones' (likelihood displacement), which this term counters.

--probe runs one training step per length on the longest example and reports peak GPU memory,
so the largest length that fits in 32 GB without spilling into shared system memory can be
picked. --offload keeps the activations saved for backward (the checkpointed layer inputs,
~0.65 GB per 1k tokens) in pinned CPU memory instead of on the GPU.
"""

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
BASE_MODEL = Path.home() / 'models' / 'gemma-4-31b-it-qat-q4_0-unquantized-5gb'
sys.path.insert(0, str(PROJECT / 'finetune'))
# Above this much reserved GPU memory (GB = 1e9 bytes) the 5090 (32 GiB = 34.2e9 bytes) spills into
# shared system memory under WSL: feasibility run 3 ran 12k at 32.1 GB at full speed, while 16k at
# 36.1 GB ran 15x slower.
SPILL_GB = 33.0
# Logit chunk budget for unsloth's fused cross entropy (see the patch in main()).
CE_CHUNK_GB = 2.0


def load_examples(path: Path, max_len: int) -> list[dict]:
    examples = []
    for line in path.read_text().splitlines():
        e = json.loads(line)
        spans = [s for s in e['spans'] if s[1] <= max_len]
        if not spans:
            continue
        ids = e['input_ids'][:spans[-1][1]]
        labels = [-100] * len(ids)
        for a, b in spans:
            labels[a:b] = ids[a:b]
        examples.append({'task': e['task'], 'run': e['run'], 'input_ids': ids, 'labels': labels,
                         'n_loss': sum(b - a for a, b in spans), 'label': e.get('label')})
    return examples


def split_validation(examples: list[dict], n_tasks: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Hold out every example of a few tasks to watch for overfitting."""
    tasks = sorted({e['task'] for e in examples})
    held = set(random.Random(seed).sample(tasks, min(n_tasks, len(tasks) // 5)))
    return [e for e in examples if e['task'] not in held], [e for e in examples if e['task'] in held]


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--base', default=str(BASE_MODEL), help='trainable checkpoint (default: the 31B unquantized QAT)')
    parser.add_argument('--base-name', default='google/gemma-4-31b-it-qat-w4a16-ct', help='served model the adapter targets')
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-len', type=int, default=12288)
    parser.add_argument('--epochs', type=float, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--grad-accum', type=int, default=8)
    parser.add_argument('--val-tasks', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--offload', action='store_true')
    parser.add_argument('--probe', help='comma-separated lengths: measure memory, do not train')
    parser.add_argument('--kto', action='store_true', help='KTO on a build-kto dataset (examples carry `label`)')
    parser.add_argument('--beta', type=float, default=10.0, help='KTO: scale of the length-normalized reward')
    parser.add_argument('--lambda-d', type=float, default=1.0, help='KTO: weight of desirable examples')
    parser.add_argument('--lambda-u', type=float, default=None,
                        help='KTO: weight of undesirable examples (default: n_desirable / n_undesirable)')
    parser.add_argument('--sft-weight', type=float, default=0.0,
                        help='KTO: weight of an added mean-NLL (SFT) term on desirable examples; 0 = pure KTO')
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_file = (out / 'train_log.jsonl').open('a', buffering=1)

    def log(**row):
        row['time'] = time.strftime('%F %T')
        log_file.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    probe_lens = [int(n) for n in args.probe.split(',')] if args.probe else []
    max_len = max(probe_lens) if probe_lens else args.max_len
    examples = load_examples(Path(args.data), max_len)

    import unsloth  # noqa: F401  (must be imported before transformers)
    import torch
    import unsloth_zoo.fused_losses.cross_entropy_loss as fused_ce
    from unsloth import FastModel
    from feasibility import export_for_vllm

    # Unsloth sizes fused cross-entropy chunks from the driver's free memory, recomputed for each
    # new sequence length. Memory held in PyTorch's cache counts as used there, so once the cache
    # fills the 32 GB card it reports ~0 and raises "No or negligible GPU memory available for
    # fused cross entropy" (v1 run 1 died at step 2). A fixed chunk budget avoids that.
    fused_ce._free_target_gb = lambda: CE_CHUNK_GB

    torch.manual_seed(args.seed)
    t0 = time.time()
    model, _ = FastModel.from_pretrained(
        model_name=args.base, max_seq_length=max_len, dtype=torch.bfloat16,
        load_in_4bit=True, text_only=True, use_gradient_checkpointing='unsloth')
    model = FastModel.get_peft_model(
        model, r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
        finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        use_gradient_checkpointing='unsloth', random_state=args.seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(event='loaded', seconds=round(time.time() - t0), memory_gb=round(torch.cuda.memory_allocated() / 1e9, 2),
        trainable_m=round(sum(p.numel() for p in trainable) / 1e6, 1), examples=len(examples),
        offload=args.offload, args=vars(args))
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    def saved_tensors_context():
        return torch.autograd.graph.save_on_cpu(pin_memory=True) if args.offload else contextlib.nullcontext()

    def loss_of(example: dict):
        torch.cuda.empty_cache()  # return cached blocks so the driver's free-memory figure stays meaningful
        ids = torch.tensor([example['input_ids']], device='cuda')
        labels = torch.tensor([example['labels']], device='cuda')
        with saved_tensors_context():
            return model(input_ids=ids, labels=labels).loss

    if probe_lens:
        longest = max(examples, key=lambda e: len(e['input_ids']))
        model.train()
        for length in probe_lens:
            ids, labels = longest['input_ids'][:length], longest['labels'][:length]
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            try:
                loss = loss_of({'input_ids': ids, 'labels': labels})
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_reserved() / 1e9
                log(event='probe', length=len(ids), seconds=round(time.time() - t, 1),
                    tokens_per_second=round(len(ids) / (time.time() - t)), loss=round(loss.item(), 4),
                    peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                    peak_reserved_gb=round(peak, 2), offload=args.offload)
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                log(event='probe', length=len(ids), oom=True, offload=args.offload)
                break
            if peak > SPILL_GB:
                break  # longer ones spill too and run ~15x slower
        return

    train, val = split_validation(examples, args.val_tasks, args.seed)
    log(event='data', train=len(train), val=len(val), val_tasks=sorted({e['task'] for e in val}),
        train_loss_tokens=sum(e['n_loss'] for e in train), train_tokens=sum(len(e['input_ids']) for e in train))
    if args.kto:
        if any(e['label'] is None for e in examples):
            raise SystemExit('--kto needs a build-kto dataset: every example must carry `label`')
        n_d, n_u = sum(e['label'] for e in train), sum(not e['label'] for e in train)
        if not n_d or not n_u:
            raise SystemExit(f'--kto needs both labels in the training split ({n_d} desirable, {n_u} undesirable)')
        if args.lambda_u is None:
            args.lambda_u = args.lambda_d * n_d / n_u
        # Reference losses: the zero-initialised adapter leaves the base model unchanged, so this pass
        # (before any optimizer step) scores every example under the reference policy.
        t = time.time()
        model.eval()
        with torch.no_grad():
            for e in train + val:
                e['ref'] = loss_of(e).item()
        model.train()
        log(event='kto_reference', seconds=round(time.time() - t), desirable=n_d, undesirable=n_u,
            lambda_d=args.lambda_d, lambda_u=round(args.lambda_u, 4), beta=args.beta, sft_weight=args.sft_weight,
            ref_nll_desirable=round(sum(e['ref'] for e in train if e['label']) / n_d, 4),
            ref_nll_undesirable=round(sum(e['ref'] for e in train if not e['label']) / n_u, 4))
    kto_z = 0.0  # running estimate of the mean implied reward (KTO's reference point), kept >= 0

    def kto_loss(e: dict, policy_nll):
        r = e['ref'] - policy_nll  # length-normalized implied reward (a tensor with grad when training)
        if e['label']:
            loss = args.lambda_d * (1 - torch.sigmoid(args.beta * (r - kto_z)))
            if args.sft_weight:  # keeps the desirable chains' own likelihood rising, not just the margin
                loss = loss + args.sft_weight * policy_nll
            return loss, r
        return args.lambda_u * (1 - torch.sigmoid(args.beta * (kto_z - r))), r
    rng = random.Random(args.seed)
    order = []
    for _ in range(math.ceil(args.epochs)):
        epoch = list(range(len(train)))
        rng.shuffle(epoch)
        order.extend(epoch)
    order = order[:round(args.epochs * len(train))]
    groups = [order[i:i + args.grad_accum] for i in range(0, len(order), args.grad_accum)]
    warmup = max(1, len(groups) // 10)

    def validation_loss() -> float | dict | None:
        if not val:
            return None
        model.eval()
        total, count = 0.0, 0
        rewards = {True: [], False: []}
        with torch.no_grad():
            for e in val:
                nll = loss_of(e).item()
                total += nll * e['n_loss']
                count += e['n_loss']
                if args.kto:
                    rewards[bool(e['label'])].append(e['ref'] - nll)
        model.train()
        if not args.kto:
            return total / count
        mean = lambda xs: round(sum(xs) / len(xs), 5) if xs else None
        r_d, r_u = mean(rewards[True]), mean(rewards[False])
        return {'nll': round(total / count, 5), 'reward_desirable': r_d, 'reward_undesirable': r_u,
                'margin': round(r_d - r_u, 5) if r_d is not None and r_u is not None else None}

    log(event='validation', step=0, loss=validation_loss())
    model.train()
    seen = 0
    for step, group in enumerate(groups):
        lr = lr_at(step, len(groups), args.lr, warmup)
        for g in optimizer.param_groups:
            g['lr'] = lr
        n_group = sum(train[i]['n_loss'] for i in group)
        t = time.time()
        torch.cuda.reset_peak_memory_stats()
        total = 0.0
        group_rewards = {True: [], False: []}
        for i in group:
            e = train[i]
            loss = loss_of(e)
            if args.kto:  # every example weighs the same: the reward is already length-normalized
                kloss, r = kto_loss(e, loss)
                (kloss / len(group)).backward()
                total += kloss.item() * n_group / len(group)  # so total / n_group below is the mean KTO loss
                group_rewards[bool(e['label'])].append(r.item())
            else:
                (loss * e['n_loss'] / n_group).backward()
                total += loss.item() * e['n_loss']
        if args.kto:
            batch = group_rewards[True] + group_rewards[False]
            kto_z = max(0.0, 0.9 * kto_z + 0.1 * sum(batch) / len(batch))
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens = sum(len(train[i]['input_ids']) for i in group)
        seen += len(group)
        extra = {}
        if args.kto:
            mean = lambda xs: round(sum(xs) / len(xs), 5) if xs else None
            extra = {'reward_desirable': mean(group_rewards[True]), 'reward_undesirable': mean(group_rewards[False]),
                     'z': round(kto_z, 5)}
        log(event='step', step=step + 1, of=len(groups), epoch=round(seen / len(train), 2), lr=lr,
            loss=round(total / n_group, 4), grad_norm=round(grad_norm, 3), seconds=round(time.time() - t, 1),
            tokens_per_second=round(tokens / (time.time() - t)),
            peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2), **extra)
        end_of_epoch = math.floor(seen / len(train)) > math.floor((seen - len(group)) / len(train))
        if end_of_epoch or step == len(groups) - 1:
            name = f'epoch{math.floor(seen / len(train))}' if end_of_epoch else 'final'
            log(event='validation', step=step + 1, loss=validation_loss())
            raw = out / f'adapter_raw_{name}'
            model.save_pretrained(str(raw))
            log(event='saved', adapter=str(out / f'adapter_{name}'), export=export_for_vllm(raw, out / f'adapter_{name}', args.base_name))


if __name__ == '__main__':
    main()
