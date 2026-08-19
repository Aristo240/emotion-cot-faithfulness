#!/usr/bin/env python3
"""
Conditional (length-preserving) family-wise null for the 50-direction sweep.

Reviewer objection: a FREE label permutation is the wrong null for a
length-corrected statistic, because the outcome is itself strongly
length-dependent (len(response) alone reaches AUC 0.888). Free permutation tests
the marginal H0 (y indep. of direction); what we need is the conditional one:

    H0:  y  independent of  direction  |  length

Design (see scripts/_null_design_notes.md for why not stratified permutation):

  statistic   nested LR chi2 for adding a direction to a length-only logistic model
  null A      free permutation of y                     (marginal   -- the old null)
  null B      y* ~ Bernoulli(p_hat), p_hat from y~length (conditional -- the fix)
  correction  max over all 50 directions per draw -> family-wise p

BOTH nulls are computed on the SAME statistic, so the comparison between them
isolates the effect of the null and nothing else.

Deterministic (seed 20260819). Analysis-only. Writes results/conditional_null.json.
"""
import json
from pathlib import Path

import numpy as np

SEED = 20260819
B = 2000
RIDGE = 1.0
ROOT = Path(__file__).resolve().parent.parent
P2 = ROOT / "results/phase2"
P4 = ROOT / "results/phase4/llama70b"
OUT = ROOT / "results/conditional_null.json"


def load(p):
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def fit_ll(X1, yy, ridge=RIDGE, iters=100, tol=1e-10):
    """Ridge-penalised logistic fit; X1 already includes the intercept column.

    The INTERCEPT IS NOT PENALISED -- penalising it shrinks fitted probabilities
    toward 0.5 and would bias any simulation drawn from this model.
    Identical to fit_pen() in scripts/paper_numbers.py; the assertion below
    verifies the two agree on the observed data.
    """
    pen = np.ones(X1.shape[1]) * ridge
    pen[0] = 0.0
    b = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = np.clip(1.0 / (1.0 + np.exp(-X1 @ b)), 1e-12, 1 - 1e-12)
        W = p * (1 - p)
        H = (X1 * W[:, None]).T @ X1 + np.diag(pen) + 1e-9 * np.eye(X1.shape[1])
        try:
            step = np.linalg.solve(H, X1.T @ (yy - p) - pen * b)
        except np.linalg.LinAlgError:
            return None, None, False
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    else:
        return b, None, False                       # hit iteration cap
    p = np.clip(1.0 / (1.0 + np.exp(-X1 @ b)), 1e-12, 1 - 1e-12)
    return b, float((yy * np.log(p) + (1 - yy) * np.log(1 - p)).sum()), True


# ------------------------------------------------------------------ dataset
# Constructed identically to paper_numbers.py: unsteered Task A only,
# labels from judge_classification.
rows = load(P4 / "extended_unsteered_judged.jsonl")
rows += [t for t in load(P2 / "task_a_judged.jsonl") if float(t.get("strength", 0)) == 0.0]
rows = [t for t in rows
        if t.get("judge_classification") in ("SHORTCUT", "LEGITIMATE") and t.get("emotion_probes")]

y = np.array([1.0 if t["judge_classification"] == "SHORTCUT" else 0.0 for t in rows])
length = np.array([len(t.get("response", "")) for t in rows], float)
zlen = (length - length.mean()) / length.std()
n, ev = len(y), int(y.sum())

emos = [e for e in sorted(rows[0]["emotion_probes"])
        if np.std([float(t["emotion_probes"][e]) for t in rows]) > 0]


def zcol(e):
    x = np.array([float(t["emotion_probes"][e]) for t in rows])
    return (x - x.mean()) / x.std()


X0 = np.column_stack([np.ones(n), zlen])
X1s = [np.column_stack([X0, zcol(e)]) for e in emos]

print(f"n = {n}, events = {ev}, directions = {len(emos)}, B = {B}, ridge = {RIDGE}")
print(f"p-value resolution floor = 1/(B+1) = {1/(B+1):.2e}"
      f"  -> claims are made at alpha=.05, not at p<1e-4")


def lr_stats(yy):
    """Nested LR chi2 for every direction. Returns (array, all_converged)."""
    _, l0, ok0 = fit_ll(X0, yy)
    if not ok0:
        return None, False
    out = np.empty(len(emos))
    ok = True
    for j, X1 in enumerate(X1s):
        _, l1, o = fit_ll(X1, yy)
        if not o:
            ok = False
            out[j] = 0.0
        else:
            out[j] = 2 * (l1 - l0)
    return out, ok


obs, obs_ok = lr_stats(y)
assert obs_ok, "observed fits did not all converge"

# ---- consistency: observed chi2 must match what paper_numbers.py reported
pn = json.load(open(ROOT / "results/paper_numbers.json"))["nested_lr"]["penalised"]
worst = max(abs(obs[emos.index(k)] - v["chi2"]) for k, v in pn.items())
print(f"\nCONSISTENCY: max |chi2 here - chi2 in paper_numbers.json| = {worst:.2e}")
assert worst < 1e-6, "the two logistic implementations disagree -- fix before trusting output"

# ------------------------------- diagnostic: why not stratified permutation
q = np.quantile(length, [0.2, 0.4, 0.6, 0.8])
strata = np.digitize(length, q)
counts = [[int((strata == s).sum()), int(y[strata == s].sum())] for s in range(5)]
frozen = sum(c for c, e in counts if e == 0 or e == c)
print(f"\nEvents per length quintile [n, events]: {counts}")
print(f"  trials in strata with zero permutation freedom: {frozen}/{n} = {frozen/n:.0%}")
print("  -> stratified permutation would be near-degenerate; parametric bootstrap used")

# ------------------------------------------------------------------- nulls
rng = np.random.default_rng(SEED)
bhat, _, _ = fit_ll(X0, y)
phat = np.clip(1.0 / (1.0 + np.exp(-X0 @ bhat)), 1e-12, 1 - 1e-12)

maxA = np.empty(B)   # free permutation (marginal)
maxB = np.empty(B)   # parametric bootstrap (conditional)
evA = np.empty(B)
evB = np.empty(B)
failA = failB = 0
for b in range(B):
    ya = rng.permutation(y)
    sa, oka = lr_stats(ya)
    evA[b] = ya.sum()
    maxA[b] = sa.max() if sa is not None else 0.0
    failA += (not oka)

    yb = (rng.random(n) < phat).astype(float)
    sb, okb = lr_stats(yb)
    evB[b] = yb.sum()
    maxB[b] = sb.max() if sb is not None else 0.0
    failB += (not okb)

pA = {e: float((np.sum(maxA >= obs[j]) + 1) / (B + 1)) for j, e in enumerate(emos)}
pB = {e: float((np.sum(maxB >= obs[j]) + 1) / (B + 1)) for j, e in enumerate(emos)}

# ------------------------------------------------------------------ self-checks
print("\nSELF-CHECKS")
print(f"  null calibration: mean simulated events, free {evA.mean():.2f} / "
      f"conditional {evB.mean():.2f}  vs observed {ev}")
print(f"  draws with a non-convergent fit: free {failA}/{B}, conditional {failB}/{B}")
print(f"  max-null chi2, 95th pct: free {np.percentile(maxA,95):.2f}, "
      f"conditional {np.percentile(maxB,95):.2f}")
stricter = sum(pB[e] >= pA[e] for e in emos)
print(f"  conditional p >= free p for {stricter}/{len(emos)} directions "
      f"-> {'conditional null is stricter, as expected' if stricter > len(emos)//2 else 'WARNING: conditional null is LOOSER'}")
nA, nB = sum(v < .05 for v in pA.values()), sum(v < .05 for v in pB.values())
print(f"  survivors at alpha=.05: free {nA}/50 -> conditional {nB}/50")

# --------------------------------------------------------------------- table
order = sorted(emos, key=lambda e: (pB[e], -obs[emos.index(e)]))
print(f"\n{'direction':<16}{'chi2':>8}{'free p':>10}{'cond. p':>10}")
for e in order[:10]:
    j = emos.index(e)
    print(f"{e:<16}{obs[j]:>8.2f}{pA[e]:>10.4f}{pB[e]:>10.4f}"
          f"{'   SURVIVES' if pB[e] < .05 else ''}")
j = emos.index("desperate")
print(f"{'desperate*':<16}{obs[j]:>8.2f}{pA['desperate']:>10.4f}{pB['desperate']:>10.4f}"
      f"   * PREREGISTERED")

# ------------------------------------------------- are the survivors independent?
# 17 correlated directions are not 17 findings. Report the effective dimensionality
# of the surviving set so the count cannot be read as 17 discoveries.
surv = [e for e in order if pB[e] < .05]
Zs = np.column_stack([zcol(e) for e in surv])
C = np.corrcoef(Zs.T)
offdiag = C[np.triu_indices_from(C, 1)]
eig = np.linalg.eigvalsh(C)[::-1]
pc1 = float(eig[0] / eig.sum())
pr = float(eig.sum() ** 2 / (eig ** 2).sum())
print(f"\n  survivor set structure: mean |r| = {np.abs(offdiag).mean():.3f} "
      f"(range {offdiag.min():.2f} to {offdiag.max():.2f})")
print(f"    PC1 explains {100*pc1:.1f}% of variance; participation ratio = {pr:.2f}")
print("    -> essentially ONE axis detected many times, not independent findings")

json.dump({
    "survivor_structure": {"mean_abs_r": float(np.abs(offdiag).mean()),
                           "min_r": float(offdiag.min()), "max_r": float(offdiag.max()),
                           "pc1_var_explained": pc1, "participation_ratio": pr},
    "n": n, "events": ev, "n_directions": len(emos), "B": B, "ridge": RIDGE,
    "seed": SEED, "p_resolution_floor": 1 / (B + 1),
    "quintile_counts": counts, "frozen_fraction": frozen / n,
    "chi2_max_disagreement_vs_paper_numbers": worst,
    "mean_sim_events": {"free": float(evA.mean()), "conditional": float(evB.mean())},
    "nonconvergent_draws": {"free": failA, "conditional": failB},
    "chi2": {e: float(obs[j]) for j, e in enumerate(emos)},
    "p_free": pA, "p_conditional": pB,
    "n_survivors_free": nA, "n_survivors_conditional": nB,
    "survivors_conditional": [e for e in order if pB[e] < .05],
    "n_conditional_ge_free": stricter,
}, open(OUT, "w"), indent=1)
print(f"\nwritten: {OUT.relative_to(ROOT)}")
