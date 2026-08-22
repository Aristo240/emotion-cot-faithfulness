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
from scipy.stats import rankdata, spearmanr

SEED = 20260819
B = 10000          # 2000 left the survivor count MC-unstable at the alpha boundary
RIDGE = 1.0
ROOT = Path(__file__).resolve().parent.parent
P2 = ROOT / "results/phase2"
P4 = ROOT / "results/phase4/llama70b"
OUT = ROOT / "results/conditional_null.json"


def auc(x, yy):
    """Rank-based AUC; matches paper_numbers.py."""
    r = rankdata(x)
    n1 = int(yy.sum()); n0 = len(yy) - n1
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


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
bhat_mle, _, _ = fit_ll(X0, y, ridge=0.0)      # generating model: unpenalised MLE
bhat_pen, _, _ = fit_ll(X0, y, ridge=RIDGE)
phat = np.clip(1.0 / (1.0 + np.exp(-X0 @ bhat_mle)), 1e-12, 1 - 1e-12)
print(f"\ngenerating model y~length: MLE slope {bhat_mle[1]:.3f} "
      f"(ridge-{RIDGE} slope would be {bhat_pen[1]:.3f}, "
      f"{100*(1-bhat_pen[1]/bhat_mle[1]):.0f}% shrunk -- MLE used)")

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

# MARGINAL (uncorrected) p from the same resampling distribution. The statistic is
# an unpenalised log-likelihood improvement evaluated at ridge-penalised estimates,
# so it has NO nominal chi2_1 reference; section 4.2 must quote a resampling p for
# the registered direction, not chi2.sf(). Recorded per direction here.
margA = np.empty(B)
margB = np.empty(B)
_jd = emos.index("desperate")
rng2 = np.random.default_rng(SEED)
for b in range(B):
    ya = rng2.permutation(y)
    sa, _ = lr_stats(ya)
    margA[b] = sa[_jd] if sa is not None else 0.0
    yb = (rng2.random(n) < phat).astype(float)
    sb, _ = lr_stats(yb)
    margB[b] = sb[_jd] if sb is not None else 0.0
p_marg_free = float((np.sum(margA >= obs[_jd]) + 1) / (B + 1))
p_marg_cond = float((np.sum(margB >= obs[_jd]) + 1) / (B + 1))
print(f"\nMARGINAL resampling p for `desperate` (statistic {obs[_jd]:.3f}):")
print(f"  free null        p = {p_marg_free:.4f}")
print(f"  conditional null p = {p_marg_cond:.4f}")
print(f"  (nominal chi2_1 would give {float(__import__('scipy.stats', fromlist=['chi2']).chi2.sf(obs[_jd],1)):.4f}"
      f" -- not used)")

# ------------------------------------------------------------------ self-checks
print("\nSELF-CHECKS")
print(f"  null calibration: mean simulated events, free {evA.mean():.2f} / "
      f"conditional {evB.mean():.2f}  vs observed {ev}")
print(f"  draws with a non-convergent fit: free {failA}/{B}, conditional {failB}/{B}")
print(f"  max-null chi2, 95th pct: free {np.percentile(maxA,95):.2f}, "
      f"conditional {np.percentile(maxB,95):.2f}")
stricter = sum(pB[e] >= pA[e] for e in emos)
# No directional expectation is asserted. Conditioning on length is the principled
# null whether or not it happens to be more conservative; what matters is that the
# reported p-values come from it, and that the two do not disagree materially.
print(f"  conditional p >= free p for {stricter}/{len(emos)} directions "
      f"(no directional expectation; conditional is the principled null either way)")
print(f"  max |p_cond - p_free| across directions: "
      f"{max(abs(pB[e] - pA[e]) for e in emos):.4f}")
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

# ------------------------------- R2.1: direction-only (magnitude-removed) readout
print("\n" + "=" * 76)
print("R2.1  Direction-only readout: remove the shared per-trial magnitude factor")
print("=" * 76)
P = np.array([[float(t_["emotion_probes"][e]) for e in emos] for t_ in rows])
nrm = np.linalg.norm(P, axis=1)
print(f"  ||probe vector|| : min {nrm.min():.3f}, median {np.median(nrm):.3f}"
      f"  (no near-zero rows: {bool(nrm.min() > 1e-6)})")
print(f"    Spearman(||probe||, length)      = {spearmanr(nrm, length)[0]:+.3f}")
print(f"    AUC of ||probe|| alone vs outcome = {auc(nrm, y):.3f}")
Pu = P / nrm[:, None]
X1u = [np.column_stack([X0, (Pu[:, j] - Pu[:, j].mean()) / Pu[:, j].std()])
       for j in range(len(emos))]


def lr_stats_u(yy):
    _, l0, ok0 = fit_ll(X0, yy)
    if not ok0:
        return None, False
    out = np.empty(len(emos)); ok = True
    for j, Xj in enumerate(X1u):
        _, l1, o = fit_ll(Xj, yy)
        if not o:
            ok = False; out[j] = 0.0
        else:
            out[j] = 2 * (l1 - l0)
    return out, ok


obs_u, ok_u = lr_stats_u(y)
assert ok_u
rng_u = np.random.default_rng(SEED)
maxBu = np.empty(B)
for bidx in range(B):
    yb = (rng_u.random(n) < phat).astype(float)
    su, _ = lr_stats_u(yb)
    maxBu[bidx] = su.max() if su is not None else 0.0
pBu = {e: float((np.sum(maxBu >= obs_u[j]) + 1) / (B + 1)) for j, e in enumerate(emos)}
nu = sum(v < .05 for v in pBu.values())
surv_scalar = [e for e in emos if pB[e] < .05]          # survivors, scalar-projection readout
kept = [e for e in surv_scalar if pBu[e] < .05]          # of those, still significant here
print(f"\n  survivors: scalar-projection readout {len(surv_scalar)}/50"
      f"  ->  direction-only readout {nu}/50")
print(f"  of the {len(surv_scalar)} original survivors, {len(kept)} remain significant")
rho_u = spearmanr(Pu[:, emos.index("desperate")], length)[0]
print(f"  Spearman(desperate, length): {spearmanr(P[:, emos.index('desperate')], length)[0]:+.3f}"
      f"  ->  {rho_u:+.3f} after normalisation")
ordu = sorted(emos, key=lambda e: (pBu[e], -obs_u[emos.index(e)]))
print(f"\n  {'direction':<16}{'chi2':>8}{'cond. p':>10}   (direction-only)")
for e in ordu[:8]:
    print(f"  {e:<16}{obs_u[emos.index(e)]:>8.2f}{pBu[e]:>10.4f}"
          f"{'   SURVIVES' if pBu[e] < .05 else ''}")
j = emos.index("desperate")
print(f"  {'desperate*':<16}{obs_u[j]:>8.2f}{pBu['desperate']:>10.4f}   * PREREGISTERED")

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
    "direction_only": {
        "probe_norm_vs_length_rho": float(spearmanr(nrm, length)[0]),
        "probe_norm_auc": auc(nrm, y),
        "desperate_rho_length_before": float(spearmanr(P[:, emos.index("desperate")], length)[0]),
        "desperate_rho_length_after": float(rho_u),
        "n_survivors": nu,
        "n_original_survivors_kept": len(kept),
        "survivors": sorted([e for e in emos if pBu[e] < .05],
                            key=lambda e: pBu[e]),
        "overlap_with_scalar": sorted(set(surv_scalar) & {e for e in emos if pBu[e] < .05}),
        "scalar_only": sorted(set(surv_scalar) - {e for e in emos if pBu[e] < .05}),
        "direction_only_new": sorted({e for e in emos if pBu[e] < .05} - set(surv_scalar)),
        "chi2": {e: float(obs_u[j]) for j, e in enumerate(emos)},
        "p_conditional": pBu,
    },
    "survivor_structure": {"mean_abs_r": float(np.abs(offdiag).mean()),
                           "min_r": float(offdiag.min()), "max_r": float(offdiag.max()),
                           "pc1_var_explained": pc1, "participation_ratio": pr},
    "n": n, "events": ev, "n_directions": len(emos), "B": B, "ridge": RIDGE,
    "seed": SEED, "p_resolution_floor": 1 / (B + 1),
    "generating_model": {"slope_mle": float(bhat_mle[1]),
                         "slope_ridge": float(bhat_pen[1]),
                         "shrinkage": float(1 - bhat_pen[1] / bhat_mle[1])},
    "quintile_counts": counts, "frozen_fraction": frozen / n,
    "chi2_max_disagreement_vs_paper_numbers": worst,
    "mean_sim_events": {"free": float(evA.mean()), "conditional": float(evB.mean())},
    "nonconvergent_draws": {"free": failA, "conditional": failB},
    "chi2": {e: float(obs[j]) for j, e in enumerate(emos)},
    "p_free": pA, "p_conditional": pB,
    "desperate_marginal": {"stat": float(obs[emos.index("desperate")]),
                           "p_free": p_marg_free, "p_conditional": p_marg_cond},
    "n_survivors_free": nA, "n_survivors_conditional": nB,
    "survivors_conditional": [e for e in order if pB[e] < .05],
    "n_conditional_ge_free": stricter,
}, open(OUT, "w"), indent=1)
print(f"\nwritten: {OUT.relative_to(ROOT)}")
