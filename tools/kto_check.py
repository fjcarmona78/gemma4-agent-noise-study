"""Numerical check of finetune/train.py's KTO loss against the standard formulation (Ethayarajh et al. 2024;
TRL's KTOTrainer), on synthetic per-token log-probs. The two intentional deviations in train.py are:

  1. the implied reward is LENGTH-NORMALIZED: r = mean_t(ref NLL) - mean_t(policy NLL) over the supervised
     tokens, instead of the summed log-ratio  sum_t(log pi - log pi_ref);
  2. the reference point z is a running, clamped mean of the batch rewards, instead of the clamped KL estimate
     on mismatched (x, y') pairs.

This prints both losses and gradients for the same examples so the size of each deviation is visible.
    python tools/kto_check.py
"""
import math
import random

random.seed(0)
beta_ours = 10.0


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def ours(ref_nll_mean, pol_nll_mean, label, z, lam):
    r = ref_nll_mean - pol_nll_mean
    return lam * (1 - sigmoid(beta_ours * (r - z))) if label else lam * (1 - sigmoid(beta_ours * (z - r))), r


def standard(ref_logp_sum, pol_logp_sum, label, z0, lam, beta=0.1):
    r = beta * (pol_logp_sum - ref_logp_sum)
    return lam * (1 - sigmoid(r - z0)) if label else lam * (1 - sigmoid(z0 - r)), r


print('example  len  label | ours: r(mean)  loss  dloss/dr | standard(beta=0.1): r(sum)  loss  dloss/dr')
examples = []
for i in range(6):
    n = random.choice([400, 1500, 4000])          # supervised tokens
    ref = [random.uniform(0.05, 0.6) for _ in range(n)]  # per-token NLL under the reference
    delta = random.uniform(-0.03, 0.03)            # policy drift per token after some training
    pol = [max(0.001, x + delta) for x in ref]
    label = i % 2 == 0
    examples.append((n, ref, pol, label))
z_ours = max(0.0, sum((sum(r) / len(r) - sum(p) / len(p)) for n, r, p, l in examples) / len(examples))
z_std = max(0.0, sum(0.1 * (-sum(p) + sum(r)) for n, r, p, l in examples) / len(examples))
for n, ref, pol, label in examples:
    lo, r1 = ours(sum(ref) / n, sum(pol) / n, label, z_ours, 1.0)
    ls, r2 = standard(-sum(ref), -sum(pol), label, z_std, 1.0)
    g1 = beta_ours * sigmoid(beta_ours * (r1 - z_ours)) * (1 - sigmoid(beta_ours * (r1 - z_ours)))
    g2 = sigmoid(r2 - z_std) * (1 - sigmoid(r2 - z_std))
    print(f'{n:7d}  {str(label):5s} | {r1:+.4f}  {lo:.3f}  {g1:.3f}      | {r2:+8.2f}  {ls:.3f}  {g2:.3f}')
print(f'\nreference points: ours z = {z_ours:.4f} (mean reward), standard z0 = {z_std:.4f} (KL estimate)')
print('Reading: the standard reward scales with sequence LENGTH (sum of log-ratios), so a 4,000-token chain gets a reward')
print('10x a 400-token chain for the same per-token drift and its sigmoid saturates (gradient -> 0). The length-normalized')
print('reward keeps every example in the sigmoid\'s active range, which is why beta is 10 instead of 0.1. The reference')
print('point differs in kind (batch mean of rewards vs KL on mismatched pairs); both are clamped at 0.')
