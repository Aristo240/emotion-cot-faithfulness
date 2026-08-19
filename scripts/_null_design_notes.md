# Design note: choosing the null for the length-corrected direction sweep

## What must be tested

Reviewer point A2: the max-T permutation null in `paper_numbers.py` §4.3 permutes
labels *freely*. But `len(response)` alone reaches AUC 0.888, so the outcome is
strongly length-dependent. A free permutation destroys that dependence, which
makes it an anti-conservative null for a statistic that is only *linearly*
decorrelated from length.

Formally, the hypothesis we want is conditional independence:

    H0:  y  ⟂  direction  |  length

The free permutation tests the *marginal* H0: y ⟂ direction. Wrong null.

## Option 1 — stratified permutation (rejected)

Permute labels within length strata. Preserves y–length by construction.

**Why rejected:** length nearly separates the outcome (AUC 0.888 with 14 events
in 120 trials). Strata therefore contain almost all events or almost none, and
within-stratum permutation has close to zero freedom. The test would be
over-conservative to the point of being uninformative — a null result would
reflect absent permutation entropy, not absent signal. The diagnostic below
reports events per length quintile so this is documented rather than asserted.

## Option 2 — parametric bootstrap max-T on the nested LR (adopted)

1. Fit the null model `logit(p) = a + b·zlen` (ridge-penalized, as elsewhere).
2. Simulate `y* ~ Bernoulli(p̂)` — this reproduces the y–length relationship
   *exactly*, because y* is generated from it.
3. For each simulation, compute the nested LR χ² for all 50 directions and
   record the maximum.
4. `p_maxT(e) = P(max χ²* ≥ χ²_obs(e))`.

This simultaneously fixes both reviewer points:

- **A2**: the null preserves y–length, so the test is genuinely conditional.
- **A1**: the correction is applied to the *LR statistic itself* across all 50
  directions, so the χ² column stops being post-selection inference and becomes
  family-wise corrected in its own right.

It also removes a statistic rather than adding one: the bespoke "residualized
AUC" is demoted to a descriptive effect size, and inference rests entirely on the
standard nested test.

## Self-checks the script must emit

- mean simulated event count vs the observed 14 (calibration of the null model)
- events per length quintile (documents why Option 1 was rejected)
- fraction of simulations where any fit failed to converge
- agreement between the new corrected p-values and the old free-permutation ones
  (if the new ones are uniformly smaller, the null is not doing its job)
