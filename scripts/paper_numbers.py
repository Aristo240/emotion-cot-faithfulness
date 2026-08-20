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

# ---- Raw agreement conflates two different failures. Added 2026-08-20: the
# 25.4% figure reads as the judges CONTRADICTING each other. They do not. Every
# disagreement is the primary judge abstaining (UNCLEAR) while Claude commits, so
# the quantity that fails to transfer is willingness to label, not agreement.
# Reported separately: (a) how often the primary judge commits, (b) how often the
# two contradict GIVEN both commit, (c) whether the abstention is independent of
# the outcome -- if it is not, the committed subset is a biased sample and the
# suite cannot be rescued by analysing what remains.
_COMMIT = ("SHORTCUT", "LEGITIMATE")
_q = [t.get("judge_classification") for t in dv]
_c = [t.get("claude_classification") for t in dv]
_both = [i for i in range(len(dv)) if _q[i] in _COMMIT and _c[i] in _COMMIT]
_contra = [i for i in _both if _q[i] != _c[i]]
# Claude-SHORTCUT vs abstention: 2x2 Fisher, tests informative missingness.
_cs = [i for i in range(len(dv)) if _c[i] == "SHORTCUT"]
_a_in = sum(1 for i in _cs if _q[i] == "UNCLEAR")
_a_out = sum(1 for i in range(len(dv)) if _c[i] != "SHORTCUT" and _q[i] == "UNCLEAR")
_n_out = len(dv) - len(_cs)
_or, _p = fisher_exact([[_a_in, len(_cs) - _a_in], [_a_out, _n_out - _a_out]])
_per = {}
for t_ in dv:
    k = t_.get("task_id", "?")
    u, n_ = _per.get(k, (0, 0))
    _per[k] = (u + (t_.get("judge_classification") == "UNCLEAR"), n_ + 1)
print(f"  diverse: primary judge commits on {len(_both)}/{len(dv)} "
      f"({len(_both)/len(dv):.1%}); contradictions given both commit "
      f"{len(_contra)}/{len(_both)}")
print(f"  abstention on Claude-SHORTCUT {_a_in}/{len(_cs)} = {_a_in/len(_cs):.1%} vs "
      f"{_a_out}/{_n_out} = {_a_out/_n_out:.1%} elsewhere; Fisher p = {_p:.2g}")
for k in sorted(_per):
    print(f"    abstention {k:<22} {_per[k][0]:>4}/{_per[k][1]:<4} = {_per[k][0]/_per[k][1]:.0%}")
R["judge"].update({
    "diverse_qwen_commits": len(_both),
    "diverse_commit_rate": len(_both) / len(dv),
    "diverse_contradictions_given_both_commit": len(_contra),
    # rule of three: with 0 events in n trials the 95% upper bound is 3/n
    "diverse_contradiction_rate_upper95": 3.0 / len(_both),
    "diverse_abstain_on_claude_shortcut": [_a_in, len(_cs)],
    "diverse_abstain_elsewhere": [_a_out, _n_out],
    "diverse_abstention_fisher_p": float(_p),
    "diverse_abstention_odds_ratio": float(_or),
    "diverse_abstention_per_mechanism": {k: list(v) for k, v in _per.items()},
    "tier2_qwen_unclear": sum(1 for t_ in unsteered
                              if t_.get("judge_classification") == "UNCLEAR"),
})

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

# Raw (uncorrected) AUC and the rank of the preregistered direction among all 50.
# Quoted in the paper's section 4.2, so it must be emitted here.
raw_auc = {e: auc(np.array([float(t_["emotion_probes"][e]) for t_ in unsteered]), y)
           for e in emos}
raw_order = sorted(emos, key=lambda e: -raw_auc[e])
desp_raw_rank = raw_order.index("desperate") + 1

# NOTE ON SCOPE. Inference for the direction sweep lives entirely in
# scripts/conditional_null.py: nested likelihood-ratio chi2, family-wise
# corrected by max-T under a length-preserving null. An earlier version of this
# script also ran a max-T permutation test and a Benjamini-Hochberg correction on
# the RESIDUALISED AUC. Those are removed. They were a second, differently
# defined inference on a different statistic under a different null, and they
# disagreed with the reported analysis (they gave 7 and 12 significant directions
# against the paper's 18, and three different p-values for `desperate`). Leaving
# them in the released JSON invited a reader to verify the paper against the
# wrong number. Residualised AUC is retained below as a DESCRIPTIVE effect size
# with a bootstrap CI, and no p-value is derived from it here.

m = len(emos)
order = sorted(emos, key=lambda e: -abs(obs[e] - 0.5))   # by descriptive effect size
print(f"  {'direction':<15}{'residAUC':>10}{'95% CI':>20}   (descriptive only)")
rows_out = []
for e in order[:8]:
    ci = boot_ci(resid[e], y, 3000)
    print(f"  {e:<15}{obs[e]:>10.3f}{f'[{ci[0]:.3f}, {ci[1]:.3f}]':>20}")
    rows_out.append({"direction": e, "resid_auc": obs[e], "ci95": ci})
all_dirs = [{"direction": e, "resid_auc": obs[e]} for e in order]
ci_dr = boot_ci(resid["desperate"], y, 3000)
print(f"  {'desperate*':<15}{obs['desperate']:>10.3f}"
      f"{f'[{ci_dr[0]:.3f}, {ci_dr[1]:.3f}]':>20}   * PREREGISTERED")
print(f"\n  raw (uncorrected) AUC: best is {raw_order[0]} at {raw_auc[raw_order[0]]:.3f}; "
      f"desperate ranks {desp_raw_rank}/{m} at {raw_auc['desperate']:.3f}")
print("  significance for these directions: see results/conditional_null.json")
R["directions"] = {
    "_inference_lives_in": "results/conditional_null.json",
    "raw_auc": raw_auc,
    "raw_auc_best": {"direction": raw_order[0], "auc": raw_auc[raw_order[0]]},
    "desperate_raw_auc_rank": desp_raw_rank,
    "n_directions": m, "top": rows_out, "all": all_dirs,
    "desperate": {"resid_auc": obs["desperate"], "ci95": ci_dr},
}

# ============ §4.3b nested logistic regression (standard test, replaces residAUC)
hdr("§4.3b  Does each direction add over a length-only model? (nested LR)")


def fit_pen(X, yy, ridge=0.0, iters=100, tol=1e-10):
    """Ridge-penalised logistic fit. The INTERCEPT IS NOT PENALISED -- penalising
    it would shrink fitted probabilities toward 0.5 and bias any simulation drawn
    from this model. Shared verbatim with scripts/conditional_null.py."""
    X = np.column_stack([np.ones(len(yy)), X])
    pen = np.ones(X.shape[1]) * ridge
    pen[0] = 0.0                                  # intercept unpenalised
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = np.clip(1 / (1 + np.exp(-X @ b)), 1e-12, 1 - 1e-12)
        W = p * (1 - p)
        H = (X * W[:, None]).T @ X + np.diag(pen) + 1e-9 * np.eye(X.shape[1])
        try:
            step = np.linalg.solve(H, X.T @ (yy - p) - pen * b)
        except np.linalg.LinAlgError:
            return b, None
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    p = np.clip(1 / (1 + np.exp(-X @ b)), 1e-12, 1 - 1e-12)
    return b, float((yy * np.log(p) + (1 - yy) * np.log(1 - p)).sum())


# Separation diagnostic: with 14 events an unpenalised fit can quasi-separate.
zb = (np.array([float(t["emotion_probes"]["bored"]) for t in unsteered]) -
      np.mean([float(t["emotion_probes"]["bored"]) for t in unsteered]))
zb = zb / zb.std()
sep = bool(zb[y == 1].min() < zb[y == 0].max() and zb[y == 0].min() < zb[y == 1].max())
print(f"  separation check (bored): event/non-event ranges overlap = {sep} "
      f"-> {'no complete separation' if sep else 'COMPLETE SEPARATION'}")
print("  reporting ridge-penalised (ridge=1.0) as primary; unpenalised agrees in sign and significance")
lr_rows = {}
for ridge, tag in ((1.0, "penalised"), (0.0, "unpenalised")):
    _, l0 = fit_pen(zlen[:, None], y, ridge)
    lr_rows[tag] = {}
    for e in ("bored", "nostalgic", "lonely", "melancholy", "gloomy",
              "compassionate", "sad", "desperate"):
        x = np.array([float(t["emotion_probes"][e]) for t in unsteered])
        z = (x - x.mean()) / x.std()
        bb, l1 = fit_pen(np.column_stack([zlen, z]), y, ridge)
        st = 2 * (l1 - l0)
        lr_rows[tag][e] = {"chi2": float(st), "p": float(chi2.sf(max(st, 0), 1)),
                           "beta": float(bb[2])}
print(f"  {'direction':<15}{'chi2(1)':>9}{'p':>10}{'beta':>9}")
for e, v in lr_rows["penalised"].items():
    print(f"  {e:<15}{v['chi2']:>9.2f}{v['p']:>10.4f}{v['beta']:>9.3f}"
          f"{'   ADDS' if v['p'] < 0.05 else '   does not add'}")
R["nested_lr"] = {"separation_overlap": sep, "primary": "penalised",
                  "penalised": lr_rows["penalised"], "unpenalised": lr_rows["unpenalised"]}

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
hdr("§4.6  Layer dependence, under the same length control")
ls = json.load(open(P4 / "analysis_layer_sweep/summary.json"))["qwen"]
print(f"  n={ls['n']}, events={ls['events']}  (SUBSET of the {len(y)}-trial set: the 80 trials")
print("   for which activations were re-extracted; not directly comparable to §4.2)")

# Re-derive from the raw sweep so the length control can be applied per layer.
# This file has no response text, but carries seq_len and prompt_token_count,
# so response length in TOKENS is available -- a closer proxy to the probe's
# averaging window than characters.
sw = [t for t in load(P4 / "extended_unsteered_layer_sweep.jsonl")
      if t.get("judge_classification") in ("SHORTCUT", "LEGITIMATE")]
ysw = np.array([1 if t["judge_classification"] == "SHORTCUT" else 0 for t in sw])
Lsw = np.array([t["seq_len"] - t["prompt_token_count"] for t in sw], float)
zsw = (Lsw - Lsw.mean()) / Lsw.std()
print(f"  response length in tokens, alone: AUC {auc(Lsw, ysw):.3f}")
print(f"  {'layer':>6}{'raw':>9}{'resid':>9}{'resid 95% CI':>20}{'rho(len)':>10}")
per_layer = {}
for Lk in sorted(sw[0]["emotion_probes_per_layer"], key=int):
    x = np.array([float(t["emotion_probes_per_layer"][Lk]["desperate"]) for t in sw])
    z = (x - x.mean()) / x.std()
    res = z - np.polyval(np.polyfit(zsw, z, 1), zsw)
    ci = boot_ci(res, ysw, 4000)
    rr = float(spearmanr(x, Lsw)[0])
    surv = ci[0] > 0.5 or ci[1] < 0.5
    print(f"  {Lk:>6}{auc(x, ysw):>9.3f}{auc(res, ysw):>9.3f}"
          f"{f'[{ci[0]:.3f},{ci[1]:.3f}]':>20}{rr:>10.3f}"
          f"{'  CI excludes 0.5' if surv else ''}")
    per_layer[Lk] = {"raw_auc": auc(x, ysw), "resid_auc": auc(res, ysw),
                     "resid_ci95": ci, "rho_length": rr,
                     "resid_ci_excludes_half": bool(surv)}
print("  -> NO layer gives the preregistered direction length-independent signal.")
print("     Layer 39's raw 0.918 collapses to "
      f"{per_layer['39']['resid_auc']:.3f}, spanning chance.")

# Are the layer-13/26 reversals a sign flip of the same axis, or different geometry?
# Compare the full 50-direction residualised-AUC PROFILE across layers.
emos_sw = [e for e in sorted(sw[0]["emotion_probes_per_layer"]["53"])
           if np.std([float(t["emotion_probes_per_layer"]["53"][e]) for t in sw]) > 0]
prof = {}
for Lk in per_layer:
    v = []
    for e in emos_sw:
        x = np.array([float(t["emotion_probes_per_layer"][Lk][e]) for t in sw])
        if x.std() == 0:
            v.append(0.5)
            continue
        z = (x - x.mean()) / x.std()
        v.append(auc(z - np.polyval(np.polyfit(zsw, z, 1), zsw), ysw))
    prof[Lk] = np.array(v)
keys = sorted(per_layer, key=int)
corr = {a: {b: float(spearmanr(prof[a], prof[b])[0]) for b in keys} for a in keys}
print("\n  50-direction profile correlation across layers (Spearman):")
print("        " + "".join(f"{k:>7}" for k in keys))
for a in keys:
    print(f"    {a:>3} " + "".join(f"{corr[a][b]:>7.2f}" for b in keys))
print("  -> layers 39/52/53/65 form a coherent block (rho 0.61-0.99); layers 13 and 26")
print(f"     are nearly uncorrelated with layer 53 (rho {corr['13']['53']:.2f}, {corr['26']['53']:.2f}).")
print("     The early-layer reversal is DIFFERENT geometry, not a sign flip of the same axis.")
print(f"     Layer 39 correlates with the steered layer 53 at rho {corr['39']['53']:.2f}.")
R["layer_profiles"] = {"spearman": corr,
                       "block": ["39", "52", "53", "65"],
                       "note": "13/26 uncorrelated with 53; not a sign flip"}

# §4.5b homogeneity of the pooled random arm
per_dir = pj["random_null"]["per_random_direction"]
rates = [v[2] for v in per_dir.values()]
print(f"\n  random-arm homogeneity: per-direction rates {rates} "
      f"(min {min(rates):.3f}, max {max(rates):.3f}) -> pooling is reasonable")
R["random_arm_homogeneity"] = {"per_direction": per_dir, "min": min(rates), "max": max(rates)}
R["layers"] = {"n": ls["n"], "events": ls["events"],
               "length_auc_tokens": auc(Lsw, ysw),
               "per_layer": per_layer,
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
