#!/usr/bin/env python3
"""
Robustness controls added in response to review, 2026-08-20.

Five things the manuscript asserted or implied but did not compute:

  R1  rho(direction, length) for every direction. The paper convicts the
      preregistered direction with rho = 0.61 but never reported the same
      diagnostic for the directions it keeps. Without it the asymmetry reads
      as a double standard.

  R2  Task fixed effects. Reward-hacking events are unevenly distributed over
      the four fast_sum variants (v3: 9/30, v2: 0/30), so a direction that
      merely separates VARIANTS would masquerade as one that predicts the
      outcome. This project has already been bitten by exactly this failure
      mode once ("pooled AUC inflated by task-identity separability"), so the
      sweep is re-run with the nuisance model = length + task.

  R3  Nonlinear length. The Limitations concede a linear residualisation would
      not remove a nonlinear length dependence. The sweep is re-run with the
      nuisance model = a flexible length basis (cubic + quantile knots).

  R4  A like-for-like AUC for section 4.4. The manuscript compared V_text on the
      non-modal subset against V_int on the FULL pooled set. The comparison
      the sentence needs is V_int on the SAME subset.

  R5  A confidence interval for the causal contrast. "The design excludes the
      claimed magnitude" is an interval claim; the manuscript supported it with
      a power calculation. The interval is computed here.

R2 and R3 use the paper's own inference: nested likelihood-ratio chi2 with the
same ridge-penalised estimator as Table 2, family-wise corrected by max-T over
all 50 directions under a conditional null that preserves the nuisance model.
Only the nuisance model changes, so the comparison isolates it.

Deterministic (seed 20260819). Analysis-only, no GPU, no network.
Writes results/robustness_controls.json.  Runtime ~25 min.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

SEED = 20260819
B = 10000
RIDGE = 1.0
ALPHA = 0.05
ROOT = Path(__file__).resolve().parent.parent
P2 = ROOT / "results/phase2"
P4 = ROOT / "results/phase4/llama70b"
OUT = ROOT / "results/robustness_controls.json"


def load(p):
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def auc(x, yy):
    r = rankdata(x)
    n1 = int(yy.sum()); n0 = len(yy) - n1
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_ll(X1, yy, ridge=RIDGE, iters=100, tol=1e-10):
    """Ridge-penalised logistic fit, intercept unpenalised.
    Identical to fit_ll() in scripts/conditional_null.py."""
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
        return b, None, False
    p = np.clip(1.0 / (1.0 + np.exp(-X1 @ b)), 1e-12, 1 - 1e-12)
    return b, float((yy * np.log(p) + (1 - yy) * np.log(1 - p)).sum()), True


def fitted_p(X1, yy):
    """Unpenalised fit used only to GENERATE null draws, as in conditional_null.py:
    penalising the generating model would make simulated data less
    nuisance-dependent than the observed data."""
    b, _, ok = fit_ll(X1, yy, ridge=0.0, iters=300)
    if b is None:
        raise RuntimeError("generating model failed")
    return np.clip(1.0 / (1.0 + np.exp(-X1 @ b)), 1e-9, 1 - 1e-9)


# ------------------------------------------------------------------ dataset
# Constructed identically to paper_numbers.py and conditional_null.py.
rows = load(P4 / "extended_unsteered_judged.jsonl")
rows += [t for t in load(P2 / "task_a_judged.jsonl") if float(t.get("strength", 0)) == 0.0]
rows = [t for t in rows
        if t.get("judge_classification") in ("SHORTCUT", "LEGITIMATE") and t.get("emotion_probes")]

y = np.array([1.0 if t["judge_classification"] == "SHORTCUT" else 0.0 for t in rows])
length = np.array([len(t.get("response", "")) for t in rows], float)
zlen = (length - length.mean()) / length.std()
task = np.array([t.get("task_id", "?") for t in rows])
n = len(y)
emos = [e for e in sorted(rows[0]["emotion_probes"])
        if np.std([float(t["emotion_probes"][e]) for t in rows]) > 0]


def zcol(e):
    x = np.array([float(t["emotion_probes"][e]) for t in rows])
    return (x - x.mean()) / x.std()


Z = {e: zcol(e) for e in emos}
R = {"_meta": {"seed": SEED, "B": B, "ridge": RIDGE, "n": n, "events": int(y.sum()),
               "directions": len(emos)}}

print(f"n = {n}, events = {int(y.sum())}, directions = {len(emos)}")

# ------------------------------------------------ task inventory (scope fact)
tasks = sorted(set(task.tolist()))
per_task = {t: [int(y[task == t].sum()), int((task == t).sum())] for t in tasks}
print(f"\nR0  tasks: {len(tasks)} variants of one family -> {per_task}")
R["tasks"] = {"n_variants": len(tasks), "variants": tasks, "events_per_variant": per_task,
              "single_task_family": all(t.startswith("fast_sum") for t in tasks)}

# ---------------------------------------------------- R1 rho(direction, length)
rho_len = {e: float(spearmanr(Z[e], length).statistic) for e in emos}
R["rho_length"] = rho_len
quoted = ["bored", "lonely", "nostalgic", "melancholy", "gloomy",
          "compassionate", "sad", "desperate"]
print("\nR1  rho(direction, length):")
for e in quoted:
    print(f"      {e:<15}{rho_len[e]:+.3f}")
srv = json.load(open(ROOT / "results/conditional_null.json"))
survivors = [e for e in emos if srv["p_conditional"][e] < ALPHA]
R["rho_length_summary"] = {
    "desperate": rho_len["desperate"],
    "survivor_max_abs": float(max(abs(rho_len[e]) for e in survivors)),
    "desperate_is_max_over_all": bool(
        abs(rho_len["desperate"]) == max(abs(v) for v in rho_len.values())),
    "n_survivors": len(survivors),
}
print(f"      desperate {rho_len['desperate']:+.3f} vs max |rho| over the "
      f"{len(survivors)} survivors {R['rho_length_summary']['survivor_max_abs']:.3f}")

# ------------------------------------------------- shared max-T machinery
def maxT_sweep(X0, label):
    """Nested-LR chi2 for every direction over nuisance model X0, family-wise
    corrected by max-T under a conditional null that preserves X0."""
    X1s = [np.column_stack([X0, Z[e]]) for e in emos]

    def stats(yy):
        _, l0, ok0 = fit_ll(X0, yy)
        if not ok0:
            return None, False
        out = np.empty(len(emos)); ok = True
        for j, X1 in enumerate(X1s):
            _, l1, o = fit_ll(X1, yy)
            if not o:
                ok = False; out[j] = 0.0
            else:
                out[j] = 2 * (l1 - l0)
        return out, ok

    obs, ok = stats(y)
    assert ok, f"{label}: observed fits did not all converge"
    phat = fitted_p(X0, y)
    rng = np.random.default_rng(SEED)
    ge = np.zeros(len(emos))
    bad = 0
    for b in range(B):
        ys = (rng.random(n) < phat).astype(float)
        if ys.sum() == 0 or ys.sum() == n:
            bad += 1; continue
        s, o = stats(ys)
        if not o or s is None:
            bad += 1; continue
        ge += (s.max() >= obs)
        if (b + 1) % 2000 == 0:
            print(f"      {label}: {b+1}/{B}")
    eff = B - bad
    p = {e: float((ge[j] + 1) / (eff + 1)) for j, e in enumerate(emos)}
    chi = {e: float(obs[j]) for j, e in enumerate(emos)}
    surv = sorted([e for e in emos if p[e] < ALPHA], key=lambda e: -chi[e])
    print(f"      {label}: {len(surv)} survivors at alpha={ALPHA}; "
          f"desperate chi2={chi['desperate']:.2f}, p={p['desperate']:.4f}")
    return {"chi2": chi, "p_maxT": p, "n_survivors": len(surv),
            "survivors": surv, "nonconvergent_draws": bad}


# baseline: length only. Must reproduce conditional_null.json exactly.
X_len = np.column_stack([np.ones(n), zlen])
print("\nR2/R3 baseline check (length-only nuisance model)")
base = maxT_sweep(X_len, "length")
worst = max(abs(base["chi2"][e] - srv["chi2"][e]) for e in emos)
print(f"      max |chi2 - conditional_null.json| = {worst:.2e}")
assert worst < 1e-6, "baseline chi2 disagree with conditional_null.py"
R["baseline_length"] = base
R["baseline_matches_conditional_null"] = float(worst)

# ------------------------------------------------------ R2 task fixed effects
D = np.column_stack([(task == t).astype(float) for t in tasks[1:]])
X_lt = np.column_stack([np.ones(n), zlen, D])
print("\nR2  nuisance model = length + task fixed effects")
R["task_fixed_effects"] = maxT_sweep(X_lt, "length+task")

# --------------------------------------------------------- R3 flexible length
q = np.quantile(length, [0.2, 0.4, 0.6, 0.8])
def zz(v): return (v - v.mean()) / v.std()
X_fl = np.column_stack([np.ones(n), zz(length), zz(length ** 2), zz(length ** 3)]
                       + [zz(np.clip(length - k, 0, None)) for k in q])
print("\nR3  nuisance model = flexible length (cubic + 4 quantile knots)")
R["flexible_length"] = maxT_sweep(X_fl, "flexlength")

# ------------------------------------------------------- R4 section 4.4 like-for-like
def vtext_score(v):
    return (v.get("urgency", 3) - v.get("composure", 3) + v.get("frustration", 3)) / 3.0

pooled = []
for p in [P2 / "task_a_judged.jsonl", P4 / "finegrained_judged.jsonl"]:
    pooled += [t for t in load(p)
               if t.get("judge_classification") in ("SHORTCUT", "LEGITIMATE")
               and t.get("emotion_probes") and t.get("vtext_ratings")]
yp = np.array([1 if t["judge_classification"] == "SHORTCUT" else 0 for t in pooled])
vt = np.array([vtext_score(t["vtext_ratings"]) for t in pooled])
vi = np.array([float(t["emotion_probes"]["desperate"]) for t in pooled])
mode = float(max(set(vt.tolist()), key=vt.tolist().count))
nm = vt != mode
R["vtext_like_for_like"] = {
    "n_nonmodal": int(nm.sum()), "events_nonmodal": int(yp[nm].sum()),
    "auc_vtext_nonmodal": auc(vt[nm], yp[nm]),
    "auc_vint_nonmodal": auc(vi[nm], yp[nm]),
    "auc_vint_all": auc(vi, yp), "auc_vtext_all": auc(vt, yp),
}
g = R["vtext_like_for_like"]
R["vtext_like_for_like"]["gap_on_same_trials"] = g["auc_vint_nonmodal"] - g["auc_vtext_nonmodal"]
R["vtext_like_for_like"]["gap_as_reported"] = g["auc_vint_all"] - g["auc_vtext_all"]
print(f"\nR4  on the SAME 258 non-modal trials: V_text {g['auc_vtext_nonmodal']:.3f} "
      f"vs V_int {g['auc_vint_nonmodal']:.3f}  (gap {g['auc_vint_nonmodal']-g['auc_vtext_nonmodal']:+.3f})")
print(f"      as previously reported (different denominators): "
      f"{g['auc_vtext_nonmodal']:.3f} vs {g['auc_vint_all']:.3f}")

# ------------------------------------------------------- R5 causal interval
def rr_ci(a, na, b, nb):
    p1, p2 = a / na, b / nb
    rr = p1 / p2
    se = np.sqrt(1 / a - 1 / na + 1 / b - 1 / nb)
    return rr, float(np.exp(np.log(rr) - 1.96 * se)), float(np.exp(np.log(rr) + 1.96 * se))

rr, lo, hi = rr_ci(12, 160, 13, 200)
rrb, lob, hib = rr_ci(12, 160, 14, 120)
R["causal_interval"] = {
    "emotion_vs_random": {"rr": rr, "ci95": [lo, hi]},
    "emotion_vs_baseline": {"rr": rrb, "ci95": [lob, hib]},
    "note": ("The upper bound is what the design excludes. The manuscript's 2.42x "
             "MDE is a power statement and is reported alongside, not instead."),
}
print(f"\nR5  emotion vs random: RR {rr:.2f}, 95% CI [{lo:.2f}, {hi:.2f}]")
print(f"      emotion vs baseline: RR {rrb:.2f}, 95% CI [{lob:.2f}, {hib:.2f}]")

OUT.write_text(json.dumps(R, indent=1, sort_keys=True))
print(f"\nwrote {OUT.relative_to(ROOT)}")
