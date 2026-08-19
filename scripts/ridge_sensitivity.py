#!/usr/bin/env python3
"""
R1.2: is the direction sweep sensitive to the ridge penalty?

The penalty exists only to stabilise the ALTERNATIVE fits against quasi-separation
at 14 events; it is a free parameter that was never varied. This script repeats
the conditional-null sweep at ridge in {0, 0.5, 1.0, 2.0} and reports whether the
conclusions move.

The generating model is the unpenalised MLE at every setting (see R1.1), so the
null is held fixed and only the analysis penalty varies.

Cross-asserts that ridge=1.0 reproduces results/conditional_null.json exactly.
Deterministic (seed 20260819). Writes results/ridge_sensitivity.json.
"""
import json
from pathlib import Path

import numpy as np

SEED = 20260819
B = 2000
RIDGES = [0.0, 0.5, 1.0, 2.0]
ROOT = Path(__file__).resolve().parent.parent
P2 = ROOT / "results/phase2"
P4 = ROOT / "results/phase4/llama70b"
OUT = ROOT / "results/ridge_sensitivity.json"


def load(p):
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def fit_ll(X1, yy, ridge, iters=100, tol=1e-10):
    """Ridge-penalised logistic fit; intercept never penalised.
    Identical to fit_ll in conditional_null.py / fit_pen in paper_numbers.py."""
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
            return None, False
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    else:
        return None, False
    p = np.clip(1.0 / (1.0 + np.exp(-X1 @ b)), 1e-12, 1 - 1e-12)
    return float((yy * np.log(p) + (1 - yy) * np.log(1 - p)).sum()), True


# ---- dataset: constructed identically to the other two scripts
rows = load(P4 / "extended_unsteered_judged.jsonl")
rows += [t for t in load(P2 / "task_a_judged.jsonl") if float(t.get("strength", 0)) == 0.0]
rows = [t for t in rows
        if t.get("judge_classification") in ("SHORTCUT", "LEGITIMATE") and t.get("emotion_probes")]
y = np.array([1.0 if t["judge_classification"] == "SHORTCUT" else 0.0 for t in rows])
length = np.array([len(t.get("response", "")) for t in rows], float)
zlen = (length - length.mean()) / length.std()
n = len(y)
emos = [e for e in sorted(rows[0]["emotion_probes"])
        if np.std([float(t["emotion_probes"][e]) for t in rows]) > 0]
X0 = np.column_stack([np.ones(n), zlen])
X1s = []
for e in emos:
    x = np.array([float(t["emotion_probes"][e]) for t in rows])
    X1s.append(np.column_stack([X0, (x - x.mean()) / x.std()]))

# generating model held fixed at the unpenalised MLE (R1.1)
bmle = np.zeros(2)
for _ in range(200):
    p = np.clip(1.0 / (1.0 + np.exp(-X0 @ bmle)), 1e-12, 1 - 1e-12)
    W = p * (1 - p)
    step = np.linalg.solve((X0 * W[:, None]).T @ X0 + 1e-9 * np.eye(2), X0.T @ (y - p))
    bmle = bmle + step
    if np.max(np.abs(step)) < 1e-11:
        break
phat = np.clip(1.0 / (1.0 + np.exp(-X0 @ bmle)), 1e-12, 1 - 1e-12)

print(f"n = {n}, events = {int(y.sum())}, directions = {len(emos)}, B = {B}")
print(f"generating model held fixed: unpenalised MLE slope {bmle[1]:.3f}\n")


def sweep(ridge):
    def stats(yy):
        l0, ok0 = fit_ll(X0, yy, ridge)
        if not ok0:
            return None, 0
        out = np.empty(len(emos))
        bad = 0
        for j, X1 in enumerate(X1s):
            l1, o = fit_ll(X1, yy, ridge)
            if not o:
                bad += 1
                out[j] = 0.0
            else:
                out[j] = 2 * (l1 - l0)
        return out, bad

    obs, bad_obs = stats(y)
    rng = np.random.default_rng(SEED)
    mx = np.empty(B)
    bad = bad_obs
    for b in range(B):
        yb = (rng.random(n) < phat).astype(float)
        s, nb = stats(yb)
        bad += nb
        mx[b] = s.max() if s is not None else 0.0
    p = {e: float((np.sum(mx >= obs[j]) + 1) / (B + 1)) for j, e in enumerate(emos)}
    return obs, p, mx, bad


res = {}
print(f"{'ridge':>7}{'survivors':>11}{'max chi2':>10}{'null 95th':>11}"
      f"{'desperate chi2':>16}{'desperate p':>13}{'bad fits':>10}")
for r in RIDGES:
    obs, p, mx, bad = sweep(r)
    surv = sorted([e for e in emos if p[e] < 0.05], key=lambda e: p[e])
    j = emos.index("desperate")
    print(f"{r:>7.1f}{len(surv):>11}{obs.max():>10.1f}{np.percentile(mx,95):>11.2f}"
          f"{obs[j]:>16.2f}{p['desperate']:>13.4f}{bad:>10}")
    res[str(r)] = {"n_survivors": len(surv), "survivors": surv,
                   "max_chi2": float(obs.max()),
                   "null_p95": float(np.percentile(mx, 95)),
                   "desperate_chi2": float(obs[j]), "desperate_p": p["desperate"],
                   "nonconvergent_fits": int(bad)}

# ---- consistency: ridge=1.0 must reproduce the primary analysis exactly
prim = json.load(open(ROOT / "results/conditional_null.json"))
d_chi2 = abs(res["1.0"]["desperate_chi2"] - prim["chi2"]["desperate"])
d_p = abs(res["1.0"]["desperate_p"] - prim["p_conditional"]["desperate"])
overlap = set(res["1.0"]["survivors"]) & set(prim["survivors_conditional"])
mc_se = (0.5 * 0.5 / B) ** 0.5
print(f"\nCONSISTENCY vs conditional_null.json at ridge=1.0:")
print(f"  |d chi2(desperate)| = {d_chi2:.2e}   (must be 0: deterministic given data+ridge)")
print(f"  |d p(desperate)|    = {d_p:.4f}   vs MC SE {mc_se:.4f} at this B")
print(f"  survivor overlap    = {len(overlap)}/{len(prim['survivors_conditional'])}")
print("  NOTE: p-values here come from a different draw sequence (that script also")
print("  consumes permutation variates), so only the statistic can match exactly.")
assert d_chi2 < 1e-9, "chi2 disagrees -- the two implementations differ"
assert d_p < 5 * mc_se, "p disagrees by more than 5 MC standard errors"

# ---- stability of the conclusions
base = set(res["1.0"]["survivors"])
print("\nSTABILITY")
for r in RIDGES:
    s = set(res[str(r)]["survivors"])
    print(f"  ridge {r:<4}: {len(s & base)}/{len(base)} of the ridge-1.0 survivors retained; "
          f"desperate {'NULL' if res[str(r)]['desperate_p'] >= 0.05 else 'SIGNIFICANT'}")
desp_null = all(res[str(r)]["desperate_p"] >= 0.05 for r in RIDGES)
axis_holds = all(res[str(r)]["n_survivors"] > 0 for r in RIDGES)
print(f"\n  desperate null at every ridge : {desp_null}")
print(f"  an axis survives at every ridge: {axis_holds}")

res["_meta"] = {"B": B, "seed": SEED, "ridges": RIDGES,
                "generating_slope_mle": float(bmle[1]),
                "desperate_null_at_all_ridges": bool(desp_null),
                "axis_survives_at_all_ridges": bool(axis_holds),
                "min_overlap_with_ridge1": min(
                    len(set(res[str(r)]["survivors"]) & base) for r in RIDGES)}
json.dump(res, open(OUT, "w"), indent=1)
print(f"\nwritten: {OUT.relative_to(ROOT)}")
