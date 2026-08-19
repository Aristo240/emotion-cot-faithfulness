#!/usr/bin/env python3
"""
Review diagnostics (2026-08-19). Analysis-only; no GPU, no model load.

Answers the open questions raised in the OpenReview-style self-review:
  W1  Does V_internal add anything over a trivial response-length baseline?
  W2  What effect size could the tier-4 random-direction null actually detect?
  W3  Are the steering arms below the unsteered baseline (degradation artifact)?
  W4  What produces the empty middle tertile in the V_text distribution?
  Q6  Do any released trials carry a pre-decision (assistant_header) probe?

Writes results/review_diagnostics.json and prints a report.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr, fisher_exact, norm, chi2

ROOT = Path(__file__).resolve().parent.parent
PHASE2 = ROOT / "results" / "phase2"
PHASE4 = ROOT / "results" / "phase4" / "llama70b"
OUT = ROOT / "results" / "review_diagnostics.json"

report = {}


def load(p):
    return [json.loads(line) for line in open(p)]


def auc(x, y):
    r = rankdata(x)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_auc_ci(x, y, n_boot=5000, seed=20260819):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        vals.append(auc(x[idx], y[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------- tier-2 set
rows = load(PHASE4 / "extended_unsteered_judged.jsonl")
rows += [r for r in load(PHASE2 / "task_a_judged.jsonl") if float(r.get("strength", 0)) == 0.0]
rows = [r for r in rows
        if r.get("judge_classification") in ("SHORTCUT", "LEGITIMATE") and r.get("emotion_probes")]

y = np.array([1 if r["judge_classification"] == "SHORTCUT" else 0 for r in rows])
length = np.array([len(r.get("response", "")) for r in rows], dtype=float)
desp = np.array([float(r["emotion_probes"]["desperate"]) for r in rows])

print("=" * 78)
print("W1  Trivial-baseline control on the tier-2 set")
print("=" * 78)
print(f"n = {len(rows)}, events = {int(y.sum())}")

auc_len, auc_desp = auc(length, y), auc(desp, y)
ci_len, ci_desp = boot_auc_ci(length, y), boot_auc_ci(desp, y)
rho, rho_p = spearmanr(desp, length)

print(f"  len(response)          AUC = {auc_len:.3f}  95% CI [{ci_len[0]:.3f}, {ci_len[1]:.3f}]")
print(f"  V_int[desperate]       AUC = {auc_desp:.3f}  95% CI [{ci_desp[0]:.3f}, {ci_desp[1]:.3f}]")
print(f"  Spearman(V_int, len)   rho = {rho:.3f}  p = {rho_p:.2g}")

# Paired bootstrap on the AUC difference (length - desperate)
rng = np.random.default_rng(20260819)
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
diffs = []
for _ in range(5000):
    idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
    diffs.append(auc(length[idx], y[idx]) - auc(desp[idx], y[idx]))
d_lo, d_hi = np.percentile(diffs, [2.5, 97.5])
print(f"  paired dAUC (len - desperate) = {np.mean(diffs):+.3f}  95% CI [{d_lo:+.3f}, {d_hi:+.3f}]")
print("  -> CI includes 0" if d_lo < 0 < d_hi else "  -> CI excludes 0")

# Does V_internal add over length? Nested logistic regression + LR test.
def fit_logit(X, y, iters=200):
    X = np.column_stack([np.ones(len(y)), X]) if X.ndim else X
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        W = p * (1 - p)
        try:
            b += np.linalg.solve((X * W[:, None]).T @ X + 1e-8 * np.eye(X.shape[1]),
                                 X.T @ (y - p))
        except np.linalg.LinAlgError:
            break
    p = np.clip(1 / (1 + np.exp(-X @ b)), 1e-9, 1 - 1e-9)
    ll = float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
    return b, ll


zl = (length - length.mean()) / length.std()
zd = (desp - desp.mean()) / desp.std()
_, ll_len = fit_logit(zl[:, None], y)
_, ll_both = fit_logit(np.column_stack([zl, zd]), y)
lr_stat = 2 * (ll_both - ll_len)
lr_p = float(chi2.sf(max(lr_stat, 0), 1))
print(f"  LR test, adding V_int[desperate] to a length-only model:")
print(f"     chi2(1) = {lr_stat:.3f}, p = {lr_p:.4f}"
      f"  -> {'V_int adds signal' if lr_p < 0.05 else 'V_int adds NOTHING detectable'}")

# Length-residualised probe
beta = np.polyfit(zl, zd, 1)
resid = zd - np.polyval(beta, zl)
auc_resid = auc(resid, y)
ci_resid = boot_auc_ci(resid, y)
print(f"  V_int[desperate] residualised on length: AUC = {auc_resid:.3f} "
      f"95% CI [{ci_resid[0]:.3f}, {ci_resid[1]:.3f}]")

report["W1"] = {
    "n": len(rows), "events": int(y.sum()),
    "auc_length": auc_len, "ci_length": ci_len,
    "auc_desperate": auc_desp, "ci_desperate": ci_desp,
    "spearman_desp_length": {"rho": float(rho), "p": float(rho_p)},
    "paired_dauc_len_minus_desp": {"mean": float(np.mean(diffs)), "ci95": [float(d_lo), float(d_hi)]},
    "lr_test_vint_over_length": {"chi2": float(lr_stat), "p": lr_p},
    "auc_desperate_residualised_on_length": auc_resid,
    "ci_desperate_residualised": ci_resid,
}

# ---------------------------------------------------------------- W2 power
print()
print("=" * 78)
print("W2  Power of the tier-4 random-direction null")
print("=" * 78)

n_emo, x_emo = 160, 12
n_rnd, x_rnd = 200, 13
p_rnd = x_rnd / n_rnd
alpha, power = 0.05, 0.80
za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)


def detectable_p2(p1, n1, n2):
    """Smallest p2 > p1 detectable at 80% power, two-sided alpha=.05."""
    lo, hi = p1, 0.999
    for _ in range(200):
        mid = (lo + hi) / 2
        pbar = (p1 * n1 + mid * n2) / (n1 + n2)
        se0 = math.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
        se1 = math.sqrt(p1 * (1 - p1) / n1 + mid * (1 - mid) / n2)
        if (abs(mid - p1) - za * se0) / se1 >= zb:
            hi = mid
        else:
            lo = mid
    return hi


mde_p = detectable_p2(p_rnd, n_rnd, n_emo)
print(f"  random baseline p = {p_rnd:.3f} ({x_rnd}/{n_rnd}); emotion arm n = {n_emo}")
print(f"  MDE at 80% power / alpha=.05: emotion rate must reach {mde_p:.3f} "
      f"= {mde_p / p_rnd:.2f}x the random rate")
print(f"  Sofroniew et al. report ~14x. Detectable here? "
      f"{'YES' if 14 * p_rnd > mde_p or 14 * p_rnd >= 1 else 'NO'} "
      f"(14x = {min(14 * p_rnd, 1.0):.3f}, saturates at 1.0)")
print(f"  Observed emotion rate {x_emo / n_emo:.3f} = {(x_emo / n_emo) / p_rnd:.2f}x random")
print(f"  => The null EXCLUDES effects >= {mde_p / p_rnd:.2f}x but CANNOT exclude "
      f"smaller ones.")
report["W2"] = {
    "p_random": p_rnd, "n_emotion": n_emo, "n_random": n_rnd,
    "mde_rate_at_80pct_power": mde_p,
    "mde_relative_risk": mde_p / p_rnd,
    "observed_relative_risk": (x_emo / n_emo) / p_rnd,
    "sofroniew_claimed_rr": 14,
    "can_detect_sofroniew_magnitude": bool(14 * p_rnd > mde_p or 14 * p_rnd >= 1),
}

# ---------------------------------------------------------------- W3 degradation
print()
print("=" * 78)
print("W3  Are the steering arms below the unsteered baseline?")
print("=" * 78)

base_x, base_n = 14, 120
for lab, xx, nn in [("emotion @0.3", x_emo, n_emo), ("random  @0.3", x_rnd, n_rnd)]:
    odds, p = fisher_exact([[xx, nn - xx], [base_x, base_n - base_x]])
    print(f"  {lab} ({xx}/{nn} = {xx/nn:.3f}) vs baseline ({base_x}/{base_n} = "
          f"{base_x/base_n:.3f}):  OR = {odds:.3f}, Fisher p = {p:.4f}")
    report.setdefault("W3", {})[lab.strip()] = {
        "rate": xx / nn, "baseline_rate": base_x / base_n,
        "odds_ratio": float(odds), "fisher_p": float(p)}

# Coherence proxy: response length by steering condition
rnd_rows = load(PHASE4 / "random_directions_judged.jsonl")
fine_rows = load(PHASE4 / "finegrained_judged.jsonl")
def mean_len(rs, pred):
    v = [len(r.get("response", "")) for r in rs if pred(r)]
    return (float(np.mean(v)), len(v)) if v else (float("nan"), 0)

bl, bn = mean_len(rows, lambda r: True)
# emotion @|s|>=0.3 lives in phase2 (strengths +-0.2/0.3/0.5), not in the
# fine-grained sweep (which only spans +-0.05..0.15).
emo_rows = load(PHASE2 / "task_a_judged.jsonl")
el, en = mean_len(emo_rows, lambda r: abs(float(r.get("strength", 0))) >= 0.3)
rl, rn = mean_len(rnd_rows, lambda r: True)
print(f"  mean response length -- unsteered {bl:.0f} (n={bn}) | "
      f"emotion|s|>=0.3 {el:.0f} (n={en}) | random {rl:.0f} (n={rn})")
report["W3"]["mean_response_length"] = {
    "unsteered": bl, "emotion_abs_ge_0.3": el, "random": rl}

# ---------------------------------------------------------------- W4 V_text bug
print()
print("=" * 78)
print("W4  The empty middle tertile in V_text")
print("=" * 78)

gap = json.load(open(PHASE2 / "_lambda_partial" / "analysis_today" / "faithfulness_gap_robust.json"))
print(f"  reported: {gap['vtext_distribution']}")

# Reproduce the analysis's own inputs and composite exactly:
#   vtext_score = (urgency - composure + frustration) / 3  over
#   task_a_judged.jsonl + finegrained_judged.jsonl   (see
#   scripts/analyze_faithfulness_gap_robust.py:48,52)
def vtext_score(vt_):
    return (vt_.get("urgency", 3) - vt_.get("composure", 3) + vt_.get("frustration", 3)) / 3.0


vt, comp = [], []
for path in [PHASE2 / "task_a_judged.jsonl", PHASE4 / "finegrained_judged.jsonl"]:
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        t = json.loads(line)
        if t.get("judge_classification") not in ("SHORTCUT", "LEGITIMATE"):
            continue
        if not t.get("emotion_probes") or not t.get("vtext_ratings"):
            continue
        vt.append(t["vtext_ratings"])
        comp.append(vtext_score(t["vtext_ratings"]))
comp = np.array(comp)
print(f"  reconstructed n = {len(comp)}")
for d in ("urgency", "composure", "frustration"):
    v = np.array([float(x[d]) for x in vt if d in x])
    print(f"    {d:12s} n={len(v):4d} unique={len(np.unique(v)):3d} "
          f"min={v.min():.2f} max={v.max():.2f} sd={v.std():.4f}")

q33, q66 = np.percentile(comp, 33), np.percentile(comp, 66)
lo = int((comp <= q33).sum())
mid = int(((comp > q33) & (comp <= q66)).sum())
hi = int((comp > q66).sum())
modal_share = float((comp == q33).mean())
print(f"  composite unique values: {len(np.unique(np.round(comp, 3)))}")
print(f"    p33 = {q33:.4f}, p66 = {q66:.4f}"
      f"{'   <-- IDENTICAL' if q33 == q66 else ''}")
print(f"    tertiles: low={lo} mid={mid} high={hi}   (reported "
      f"{gap['vtext_distribution']['low_third']}/"
      f"{gap['vtext_distribution']['mid_third']}/"
      f"{gap['vtext_distribution']['high_third']})")
if q33 == q66:
    print(f"    DIAGNOSIS: p33 == p66 exactly. {100*modal_share:.1f}% of trials share")
    print("    the single modal composite value, so both percentiles land on it and")
    print("    the (p33, p66] middle bin is empty BY CONSTRUCTION. This is a binning")
    print("    artifact of a near-constant score, not a property of the trials.")
    print("    Root cause: frustration is constant at 1.0 in 985/992 trials; urgency")
    print("    and composure have sd 0.24 and 0.35 on a 1-7 scale. The V_text")
    print("    instrument is non-responsive, so any low/high V_text split -- including")
    print("    the HIDDEN-quadrant analysis -- is splitting on noise.")
report["W4"] = {
    "reported_distribution": gap["vtext_distribution"],
    "reconstructed_n": len(comp),
    "composite_unique": int(len(np.unique(np.round(comp, 3)))),
    "p33": float(q33), "p66": float(q66),
    "p33_equals_p66": bool(q33 == q66),
    "modal_value_share": modal_share,
    "tertiles": {"low": lo, "mid": mid, "high": hi},
    "reproduces_reported": bool(
        lo == gap["vtext_distribution"]["low_third"]
        and mid == gap["vtext_distribution"]["mid_third"]
        and hi == gap["vtext_distribution"]["high_third"]),
}

# ---------------------------------------------------------------- Q6
print()
print("=" * 78)
print("Q6  Is a pre-decision (assistant_header) probe present in any released data?")
print("=" * 78)
found = {}
for p in sorted(ROOT.glob("results/**/*.jsonl")):
    try:
        r0 = json.loads(open(p).readline())
    except Exception:
        continue
    keys = set(r0)
    hits = [k for k in keys if "header" in k.lower() or "token_position" in k.lower()]
    if isinstance(r0.get("emotion_probes"), dict):
        hits += [k for k in r0["emotion_probes"] if "header" in str(k).lower()]
    if hits:
        found[str(p.relative_to(ROOT))] = hits
print(f"  files carrying an assistant_header / token-position probe: {len(found)}")
for k, v in found.items():
    print(f"    {k}: {v}")
if not found:
    print("  NONE. The pre-decision readout would require regenerating activations")
    print("  (GPU). It cannot be recovered from released files.")
report["Q6"] = {"files_with_header_probe": found}

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(report, open(OUT, "w"), indent=1)
print()
print(f"written: {OUT.relative_to(ROOT)}")
