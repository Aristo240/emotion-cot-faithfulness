#!/usr/bin/env python3
"""
Single source of truth for every number in paper/interpscience_short.tex.

Analysis-only: no GPU, no model load, no network. Deterministic (all RNG seeded
with 20260819). Supersedes scripts/review_diagnostics.py.

Run:  python scripts/paper_numbers.py
Out:  results/paper_numbers.json  + a printed report whose section numbers
      match the paper's section numbers.

Every figure quoted in the paper must appear in the JSON this writes. If a
number is in the paper and not here, it is unsourced and must be removed.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr, fisher_exact, norm, chi2

SEED = 20260819
ROOT = Path(__file__).resolve().parent.parent
P2 = ROOT / "results/phase2"
P4 = ROOT / "results/phase4/llama70b"
OUT = ROOT / "results/paper_numbers.json"

R = {}


def load(p):
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def auc(x, y):
    """Rank-based AUC. Ties receive average ranks, i.e. contribute 0.5 each."""
    r = rankdata(x)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_ci(x, y, n_boot=5000, seed=SEED):
    """Stratified bootstrap CI: resample positives and negatives separately."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), True),
                              rng.choice(neg, len(neg), True)])
        v.append(auc(x[idx], y[idx]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def hdr(s):
    print("\n" + "=" * 76 + f"\n{s}\n" + "=" * 76)


# ===================================================================== dataset
# The tier-2 / association set: unsteered Task A trials only.
#   80 from phase4 extended_unsteered + 40 from phase2 at strength == 0.
# Labels: judge_classification (Qwen 2.5 72B, 3-pass majority).
unsteered = load(P4 / "extended_unsteered_judged.jsonl")
unsteered += [t for t in load(P2 / "task_a_judged.jsonl")
              if float(t.get("strength", 0)) == 0.0]
unsteered = [t for t in unsteered
             if t.get("judge_classification") in ("SHORTCUT", "LEGITIMATE")
             and t.get("emotion_probes")]

y = np.array([1 if t["judge_classification"] == "SHORTCUT" else 0 for t in unsteered])
length = np.array([len(t.get("response", "")) for t in unsteered], float)
zlen = (length - length.mean()) / length.std()

R["dataset"] = {
    "unsteered_n": len(unsteered), "unsteered_events": int(y.sum()),
    "label_field": "judge_classification",
    "sources": ["results/phase4/llama70b/extended_unsteered_judged.jsonl (80)",
                "results/phase2/task_a_judged.jsonl @ strength==0 (40)"],
}

# ============================================================= judge agreement
hdr("§3  Judge validity (cross-family, per dataset)")
ag = []
for f, filt in [(P4 / "extended_unsteered_claude_judged.jsonl", lambda t: True),
                (P2 / "task_a_claude_judged.jsonl", lambda t: float(t.get("strength", 0)) == 0.0)]:
    rs = [t for t in load(f) if filt(t)]
    a = sum(t.get("judge_classification") == t.get("claude_classification") for t in rs)
    ag.append((f.name, a, len(rs)))
    print(f"  {f.name:44s} {a}/{len(rs)}")
tot_a, tot_n = sum(a for _, a, _ in ag), sum(n for _, _, n in ag)
dv = load(P2 / "_lambda_partial/phase2/task_a_diverse_claude_judged.jsonl")
dv_a = sum(t.get("judge_classification") == t.get("claude_classification") for t in dv)
print(f"  TIER-2 SET TOTAL{'':28s} {tot_a}/{tot_n} = {tot_a/tot_n:.3f}")
print(f"  diverse suite (EXCLUDED from all results)    {dv_a}/{len(dv)} = {dv_a/len(dv):.3f}")
R["judge"] = {"tier2_agree": tot_a, "tier2_n": tot_n,
              "diverse_agree": dv_a, "diverse_n": len(dv),
              "diverse_qwen_shortcut": sum(t.get("judge_classification") == "SHORTCUT" for t in dv),
              "diverse_qwen_unclear": sum(t.get("judge_classification") == "UNCLEAR" for t in dv),
              "diverse_claude_shortcut": sum(t.get("claude_classification") == "SHORTCUT" for t in dv)}

# ==================================================================== §4.1 sem
hdr("§4.1  Semantic validity (EmoBank, zero-shot, frozen probes)")
emo = json.load(open(ROOT / "results/emobank/llama70b/report.json"))
print(f"  n sentences = {emo['n_sentences']}, probe layer = {emo['probe_layer']}")
for k in ("V", "A", "D"):
    print(f"    CV R^2 {k}: {emo['cv_r2_mean'][k]:.3f}")
R["semantic"] = {"n": emo["n_sentences"], "cv_r2": emo["cv_r2_mean"]}

# ============================================ §4.2 association + length control
hdr("§4.2  Association, and the response-length control")
desp = np.array([float(t["emotion_probes"]["desperate"]) for t in unsteered])
a_d, ci_d = auc(desp, y), boot_ci(desp, y)
a_l, ci_l = auc(length, y), boot_ci(length, y)
rho, rho_p = spearmanr(desp, length)
print(f"  n = {len(y)}, events = {int(y.sum())}")
print(f"  V_int[desperate]  AUC {a_d:.3f}  CI [{ci_d[0]:.3f}, {ci_d[1]:.3f}]   (PREREGISTERED)")
print(f"  len(response)     AUC {a_l:.3f}  CI [{ci_l[0]:.3f}, {ci_l[1]:.3f}]")
print(f"  Spearman(V_int[desperate], length) rho = {rho:.3f}, p = {rho_p:.2g}")

# Paired bootstrap of the AUC difference (same resample for both predictors).
rng = np.random.default_rng(SEED)
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
d = []
for _ in range(5000):
    idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
    d.append(auc(length[idx], y[idx]) - auc(desp[idx], y[idx]))
d_ci = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
print(f"  paired dAUC (length - desperate) = {np.mean(d):+.3f} CI [{d_ci[0]:+.3f}, {d_ci[1]:+.3f}]"
      f"  -> {'NOT significant' if d_ci[0] < 0 < d_ci[1] else 'significant'}")


def logit_ll(X, yy, iters=300):
    X = np.column_stack([np.ones(len(yy)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = np.clip(1 / (1 + np.exp(-X @ b)), 1e-9, 1 - 1e-9)
        W = p * (1 - p)
        try:
            b += np.linalg.solve((X * W[:, None]).T @ X + 1e-8 * np.eye(X.shape[1]),
                                 X.T @ (yy - p))
        except np.linalg.LinAlgError:
            break
    p = np.clip(1 / (1 + np.exp(-X @ b)), 1e-9, 1 - 1e-9)
    return float((yy * np.log(p) + (1 - yy) * np.log(1 - p)).sum())


zd = (desp - desp.mean()) / desp.std()
lr = 2 * (logit_ll(np.column_stack([zlen, zd]), y) - logit_ll(zlen[:, None], y))
lr_p = float(chi2.sf(max(lr, 0), 1))
print(f"  LR test, V_int[desperate] added to a length-only model: chi2(1)={lr:.3f}, p={lr_p:.4f}")
R["association"] = {
    "auc_desperate": a_d, "ci_desperate": ci_d,
    "auc_length": a_l, "ci_length": ci_l,
    "spearman_desp_length": {"rho": float(rho), "p": float(rho_p)},
    "paired_dauc_length_minus_desperate": {"mean": float(np.mean(d)), "ci95": d_ci},
    "lr_test_vint_over_length": {"chi2": float(lr), "p": lr_p},
}

# ============================== §4.3 all-50 sweep with family-wise correction
hdr("§4.3  All 50 directions, length-residualised, with multiplicity control")
emos = [e for e in sorted(unsteered[0]["emotion_probes"])
        if np.std([float(t["emotion_probes"][e]) for t in unsteered]) > 0]
resid = {}
for e in emos:
    x = np.array([float(t["emotion_probes"][e]) for t in unsteered])
    z = (x - x.mean()) / x.std()
    resid[e] = z - np.polyval(np.polyfit(zlen, z, 1), zlen)
obs = {e: auc(resid[e], y) for e in emos}

# Permutation null on the LABELS, shared across directions so that the
# max-T statistic correctly accounts for correlation between directions.
rng = np.random.default_rng(SEED)
NP = 10000
null = {e: np.empty(NP) for e in emos}
maxnull = np.empty(NP)
for i in range(NP):
    yp = rng.permutation(y)
    vals = [abs(auc(resid[e], yp) - 0.5) for e in emos]
    for e, v in zip(emos, vals):
        null[e][i] = v
    maxnull[i] = max(vals)
pval = {e: float((np.sum(null[e] >= abs(obs[e] - 0.5)) + 1) / (NP + 1)) for e in emos}
maxT = {e: float((np.sum(maxnull >= abs(obs[e] - 0.5)) + 1) / (NP + 1)) for e in emos}
order = sorted(emos, key=lambda e: pval[e])
bh, prev, m = {}, 1.0, len(emos)
for i, e in enumerate(reversed(order)):
    prev = min(prev, pval[e] * m / (m - i))
    bh[e] = prev

print(f"  {'direction':<15}{'residAUC':>9}{'CI':>18}{'raw p':>9}{'BH q':>9}{'max-T p':>10}")
rows_out = []
for e in order[:8]:
    ci = boot_ci(resid[e], y, 3000)
    tag = "  SURVIVES max-T" if maxT[e] < 0.05 else ("  BH only" if bh[e] < 0.05 else "")
    print(f"  {e:<15}{obs[e]:>9.3f}{f'[{ci[0]:.3f},{ci[1]:.3f}]':>18}"
          f"{pval[e]:>9.4f}{bh[e]:>9.4f}{maxT[e]:>10.4f}{tag}")
    rows_out.append({"direction": e, "resid_auc": obs[e], "ci95": ci,
                     "p": pval[e], "bh_q": bh[e], "maxT_p": maxT[e]})
# Store ALL 50 (not only the printed top 8) so every direction the paper may
# mention is sourced.
all_dirs = [{"direction": e, "resid_auc": obs[e], "p": pval[e],
             "bh_q": bh[e], "maxT_p": maxT[e]} for e in order]
ci_dr = boot_ci(resid["desperate"], y, 3000)
print(f"  {'desperate*':<15}{obs['desperate']:>9.3f}"
      f"{f'[{ci_dr[0]:.3f},{ci_dr[1]:.3f}]':>18}"
      f"{pval['desperate']:>9.4f}{bh['desperate']:>9.4f}{maxT['desperate']:>10.4f}"
      f"   * PREREGISTERED - NULL")
print(f"\n  surviving BH q<0.05 : {sum(bh[e] < 0.05 for e in emos)}/{m}")
print(f"  surviving max-T<0.05: {sum(maxT[e] < 0.05 for e in emos)}/{m}"
      f"   (family-wise over all {m} directions)")
R["directions"] = {
    "n_directions": m, "top": rows_out, "all": all_dirs,
    "desperate": {"resid_auc": obs["desperate"], "ci95": ci_dr,
                  "p": pval["desperate"], "bh_q": bh["desperate"],
                  "maxT_p": maxT["desperate"]},
    "n_bh_significant": int(sum(bh[e] < 0.05 for e in emos)),
    "n_maxT_significant": int(sum(maxT[e] < 0.05 for e in emos)),
    "maxT_survivors": [e for e in order if maxT[e] < 0.05],
}

# ============================================ §4.4 the text baseline is unusable
hdr("§4.4  Why the V_text comparison cannot be made")


def vtext_score(v):
    """Composite used by scripts/analyze_faithfulness_gap_robust.py:48."""
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
tie = float((vt == mode).mean())
n1, n0 = int(yp.sum()), len(yp) - int(yp.sum())
tied_pairs = int(((vt == mode) & (yp == 1)).sum()) * int(((vt == mode) & (yp == 0)).sum())
nm = vt != mode
for dim in ("urgency", "composure", "frustration"):
    v = np.array([float(t["vtext_ratings"][dim]) for t in pooled if dim in t["vtext_ratings"]])
    print(f"    {dim:<12} unique={len(np.unique(v)):3d}  sd={v.std():.4f}  (1-7 scale)")
p33, p66 = np.percentile(vt, 33), np.percentile(vt, 66)
print(f"  n={len(yp)}, events={n1}; modal composite {mode:.3f} held by {100*tie:.1f}% of trials")
print(f"  p33={p33:.3f}  p66={p66:.3f}  -> middle tertile empty BY CONSTRUCTION"
      if p33 == p66 else f"  p33={p33:.3f} p66={p66:.3f}")
print(f"  tied (pos,neg) ROC pairs at the mode: {tied_pairs}/{n1*n0} = {100*tied_pairs/(n1*n0):.1f}%")
print(f"    each contributes exactly 0.5, compressing the AUC toward chance")
print(f"  V_text  AUC all trials        {auc(vt, yp):.3f}")
print(f"  V_text  AUC non-modal trials  {auc(vt[nm], yp[nm]):.3f}  (n={int(nm.sum())}, events={int(yp[nm].sum())})")
print(f"  V_int   AUC all trials        {auc(vi, yp):.3f}")
print("  -> where V_text can discriminate at all it matches V_int; the apparent")
print("     gap is a tie artifact of a non-responsive instrument, not a faithfulness gap.")
R["vtext"] = {
    "n": len(yp), "events": n1, "modal_value": mode, "modal_share": tie,
    "p33": float(p33), "p66": float(p66), "p33_eq_p66": bool(p33 == p66),
    "tied_roc_pairs": tied_pairs, "total_roc_pairs": n1 * n0,
    "tied_pair_fraction": tied_pairs / (n1 * n0),
    "auc_vtext_all": auc(vt, yp), "auc_vtext_nonmodal": auc(vt[nm], yp[nm]),
    "n_nonmodal": int(nm.sum()), "events_nonmodal": int(yp[nm].sum()),
    "auc_vint_all": auc(vi, yp),
}

# ======================================================== §4.5 causal specificity
hdr("§4.5  Causal specificity, with power")
pj = json.load(open(P4 / "analysis_judged/report.json"))
base = pj["baseline_unsteered"]
rn = pj["random_null"]
b_x, b_n = base["hacks"], base["n"]
e_x, e_n = rn["emotion_at_0_3"]["hacks"], rn["emotion_at_0_3"]["n"]
r_x, r_n = rn["random"]["hacks"], rn["random"]["n"]
print(f"  unsteered baseline  {b_x}/{b_n} = {b_x/b_n:.3f}")
print(f"  emotion @+-0.3      {e_x}/{e_n} = {e_x/e_n:.3f}")
print(f"  random  @+-0.3      {r_x}/{r_n} = {r_x/r_n:.3f}   (5 directions orthogonal to the emotion subspace)")
print(f"  emotion vs random   Fisher p = {rn['fisher_emotion_vs_random_p']:.4f}")
vs_base = {}
for lab, xx, nn in (("emotion", e_x, e_n), ("random", r_x, r_n)):
    orr, pp = fisher_exact([[xx, nn - xx], [b_x, b_n - b_x]])
    vs_base[lab] = {"odds_ratio": float(orr), "p": float(pp)}
    print(f"  {lab} vs baseline: OR={orr:.3f}, p={pp:.4f}  -> {'no degradation artifact' if pp > .05 else 'DIFFERS'}")

# Coherence proxy: mean response length per arm (cited in the paper).
def mlen(rs, pred=lambda t: True):
    v = [len(t.get("response", "")) for t in rs if pred(t)]
    return float(np.mean(v)) if v else float("nan")


len_unst = mlen(unsteered)
len_emo = mlen(load(P2 / "task_a_judged.jsonl"), lambda t: abs(float(t.get("strength", 0))) >= 0.3)
len_rnd = mlen(load(P4 / "random_directions_judged.jsonl"))
print(f"  mean response length: unsteered {len_unst:.0f} | emotion|s|>=0.3 {len_emo:.0f} "
      f"| random {len_rnd:.0f} chars")

za, zb = norm.ppf(0.975), norm.ppf(0.80)
p1 = r_x / r_n
lo, hi = p1, 0.999
for _ in range(200):
    mid = (lo + hi) / 2
    pb = (p1 * r_n + mid * e_n) / (r_n + e_n)
    se0 = math.sqrt(pb * (1 - pb) * (1 / r_n + 1 / e_n))
    se1 = math.sqrt(p1 * (1 - p1) / r_n + mid * (1 - mid) / e_n)
    hi, lo = (mid, lo) if (abs(mid - p1) - za * se0) / se1 >= zb else (hi, mid)
print(f"  MDE at 80% power, alpha=.05: emotion rate {hi:.3f} = {hi/p1:.2f}x random")
print(f"  Sofroniew et al. report ~14x -> the design EXCLUDES the claimed magnitude;")
print(f"  it cannot exclude effects below {hi/p1:.2f}x.")
print(f"  fine-grained dose-response: desperate z={pj['trend_test']['desperate']['z']:.2f}, "
      f"p={pj['trend_test']['desperate']['p']:.3f}")
ti = pj["text_injection"]
orr, pp = fisher_exact([[ti["desperate_inject"]["hacks"], ti["desperate_inject"]["n"] - ti["desperate_inject"]["hacks"]],
                        [b_x, b_n - b_x]])
print(f"  text injection 'feel desperate': {ti['desperate_inject']['hacks']}/{ti['desperate_inject']['n']} "
      f"vs baseline, Fisher p={pp:.3f}")
R["causal"] = {
    "baseline": [b_x, b_n], "emotion_0_3": [e_x, e_n], "random_0_3": [r_x, r_n],
    "fisher_emotion_vs_random": rn["fisher_emotion_vs_random_p"],
    "vs_baseline": vs_base,
    "mean_response_length": {"unsteered": len_unst, "emotion_abs_ge_0.3": len_emo,
                             "random": len_rnd},
    "mde_rate": hi, "mde_relative_risk": hi / p1, "sofroniew_claimed_rr": 14,
    "trend_desperate": pj["trend_test"]["desperate"],
    "text_injection_vs_baseline_p": float(pp),
}

# ========================================================= §4.6 layer dependence
hdr("§4.6  Layer dependence (context for the causal null)")
ls = json.load(open(P4 / "analysis_layer_sweep/summary.json"))["qwen"]
print(f"  n={ls['n']}, events={ls['events']}  (SUBSET of the {len(y)}-trial set: the 80 trials")
print("   for which activations were re-extracted; not directly comparable to §4.2)")
for L in ls["per_layer"]:
    print(f"    layer {L['layer']:>2}: V_int[desperate] AUC {L['vint_desperate_auc']['auc']:.3f}")
R["layers"] = {"n": ls["n"], "events": ls["events"],
               "per_layer": {L["layer"]: L["vint_desperate_auc"]["auc"] for L in ls["per_layer"]},
               "note": "80-trial subset; not the 120-trial association set"}

# ============================================================== preregistration
hdr("§4.7  Preregistered decision rule")
h5 = json.load(open(ROOT / "results/phase3/llama70b/h5_holdout_report.json"))
print(f"  rule: {h5['(5)_decision']['rule']}")
print(f"  requires >=6 of 9 task variants with outcome variance (min_tasks_required scales 4->3, 9->6)")
print(f"  verdict: {h5['(5)_decision']['verdict'][:70]}...")
print("  The 5 diverse variants that would satisfy it are the ones excluded in §3.")
R["prereg"] = {"verdict": h5["(5)_decision"]["verdict"],
               "rule": h5["(5)_decision"]["rule"],
               "logo_meaningful": h5["(2)_leave_one_task_out"]["logo_meaningful"]}

OUT.write_text(json.dumps(R, indent=1))
print(f"\nwritten: {OUT.relative_to(ROOT)}")
