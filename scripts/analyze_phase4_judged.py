"""Phase 4 analysis on the JUDGE-VERIFIED jsonls (post-Lambda run).

Same statistics as analyze_phase4.py but uses judge_classification (Qwen 2.5 72B,
3-pass majority) instead of regex `classification`. Outputs go to
results/phase4/llama70b/analysis_judged/ and plots to results/plots/phase4j_*.png
so the regex-based earlier outputs remain side-by-side for comparison.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path("/Users/naamarozen/Desktop/Naama/Projects AI/emotion-cot-faithfulness/emotion-cot-faithfulness_16Apr")
PHASE4 = ROOT / "results/phase4/llama70b"
PHASE2 = ROOT / "results/phase2"
OUT = PHASE4 / "analysis_judged"
PLOTS = ROOT / "results/plots"
OUT.mkdir(exist_ok=True, parents=True)
PLOTS.mkdir(exist_ok=True, parents=True)
RNG = np.random.default_rng(20260508)


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in open(p) if line.strip()]


def judge_hack(d: dict) -> int:
    jc = (d.get("judge_classification") or "").upper()
    return 1 if "SHORTCUT" in jc or "HACK" in jc else 0


def proportion_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# Load JUDGED files
fg = load_jsonl(PHASE4 / "finegrained_judged.jsonl")
ext = load_jsonl(PHASE4 / "extended_unsteered_judged.jsonl")
rnd = load_jsonl(PHASE4 / "random_directions_judged.jsonl")
inj = load_jsonl(PHASE4 / "text_injection_judged.jsonl")
phase2_a = load_jsonl(PHASE2 / "task_a_judged.jsonl")
phase2_a = [d for d in phase2_a if str(d.get("task_id", "")).startswith("fast_sum")]

print("Loaded:")
print(f"  fine-grained_judged:        {len(fg)}")
print(f"  extended_unsteered_judged:  {len(ext)}")
print(f"  random_directions_judged:   {len(rnd)}")
print(f"  text_injection_judged:      {len(inj)}")
print(f"  phase2 task A (judged):     {len(phase2_a)}")


# -----------------------------------------------------------------------------
# Combined unsteered baseline (Phase 2 + Phase 4 extended)
# -----------------------------------------------------------------------------
def baseline():
    h2 = sum(judge_hack(d) for d in phase2_a if d["emotion"] == "none")
    n2 = sum(1 for d in phase2_a if d["emotion"] == "none")
    h4 = sum(judge_hack(d) for d in ext)
    n4 = len(ext)
    return h2 + h4, n2 + n4


h_base, n_base = baseline()
p_base = h_base / n_base
print(f"\nUnsteered baseline (Phase 2 + extended): {h_base}/{n_base} = {p_base:.3f}")


# -----------------------------------------------------------------------------
# Dose-response: Phase 2 wide grid + Phase 4 fine grid, judge labels throughout
# -----------------------------------------------------------------------------
def cell_counts(records, em, s):
    sub = [d for d in records if d["emotion"] == em and abs(d["strength"] - s) < 1e-6]
    return sum(judge_hack(d) for d in sub), len(sub)


strengths_grid = [-0.50, -0.30, -0.20, -0.15, -0.10, -0.05,
                  0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
dose = []
for em in ["desperate", "calm"]:
    for s in strengths_grid:
        h, n = cell_counts(phase2_a, em, s)
        if n == 0:
            h, n = cell_counts(fg, em, s)
        if n == 0:
            continue
        lo, hi = proportion_ci(h, n)
        odds, p = stats.fisher_exact([[h, n - h], [h_base, n_base - h_base]])
        dose.append(dict(emotion=em, strength=s, hacks=h, n=n,
                         rate=h / n, ci_low=lo, ci_high=hi,
                         fisher_or=float(odds), fisher_p=float(p)))
        print(f"  {em:>9s} @ {s:+.2f}: {h:>2d}/{n:>2d} = {h/n:.3f}  "
              f"[{lo:.3f}, {hi:.3f}]   p={p:.3f}")
dose_df = pd.DataFrame(dose)


# Cochran-Armitage trend
def cochran_armitage(rows):
    s = rows["strength"].values.astype(float)
    h = rows["hacks"].values.astype(float)
    n = rows["n"].values.astype(float)
    p_total = h.sum() / n.sum()
    num = ((s - s.mean()) * (h - n * p_total)).sum()
    var = p_total * (1 - p_total) * (n * (s - s.mean()) ** 2).sum()
    z = num / np.sqrt(var) if var > 0 else 0.0
    return dict(z=float(z), p=float(2 * (1 - stats.norm.cdf(abs(z)))),
                slope_sign=int(np.sign(num)))


desp_rows = dose_df[dose_df.emotion == "desperate"]
calm_rows = dose_df[dose_df.emotion == "calm"]
ca_desp = cochran_armitage(desp_rows)
ca_calm = cochran_armitage(calm_rows)
print(f"\nCochran-Armitage trend (desperate): z={ca_desp['z']:.3f}, p={ca_desp['p']:.4f}")
print(f"Cochran-Armitage trend (calm):      z={ca_calm['z']:.3f}, p={ca_calm['p']:.4f}")


# -----------------------------------------------------------------------------
# Random-direction null vs emotion at |s|=0.3
# -----------------------------------------------------------------------------
emo_at_03 = [d for d in phase2_a if d["emotion"] in ("desperate", "calm")
             and abs(abs(d["strength"]) - 0.3) < 1e-6]
h_rnd, n_rnd = sum(judge_hack(d) for d in rnd), len(rnd)
h_emo, n_emo = sum(judge_hack(d) for d in emo_at_03), len(emo_at_03)
print(f"\nRandom-direction null at |s|=0.3:")
print(f"  Random:          {h_rnd}/{n_rnd} = {h_rnd/n_rnd:.3f}")
print(f"  Emotion @|0.3|:  {h_emo}/{n_emo} = {h_emo/n_emo:.3f}")
print(f"  Baseline:        {h_base}/{n_base} = {p_base:.3f}")
odds, p_emo_vs_rnd = stats.fisher_exact([[h_emo, n_emo - h_emo],
                                         [h_rnd, n_rnd - h_rnd]])
odds_b, p_rnd_vs_b = stats.fisher_exact([[h_rnd, n_rnd - h_rnd],
                                         [h_base, n_base - h_base]])
print(f"  Fisher emotion vs random: OR={odds:.2f}, p={p_emo_vs_rnd:.4f}")
print(f"  Fisher random vs baseline: OR={odds_b:.2f}, p={p_rnd_vs_b:.4f}")
rnd_per = {}
for em in sorted({d["emotion"] for d in rnd}):
    sub = [d for d in rnd if d["emotion"] == em]
    h = sum(judge_hack(d) for d in sub)
    rnd_per[em] = (h, len(sub), h / len(sub))


# -----------------------------------------------------------------------------
# Text injection vs activation
# -----------------------------------------------------------------------------
inj_des = [d for d in inj if d["emotion"] == "desperate_inject"]
inj_cal = [d for d in inj if d["emotion"] == "calm_inject"]
h_id, n_id = sum(judge_hack(d) for d in inj_des), len(inj_des)
h_ic, n_ic = sum(judge_hack(d) for d in inj_cal), len(inj_cal)

desp_p2 = [d for d in phase2_a if d["emotion"] == "desperate" and abs(d["strength"] - 0.20) < 1e-6]
calm_p2 = [d for d in phase2_a if d["emotion"] == "calm" and abs(d["strength"] - 0.20) < 1e-6]
h_dp, n_dp = sum(judge_hack(d) for d in desp_p2), len(desp_p2)
h_cp, n_cp = sum(judge_hack(d) for d in calm_p2), len(calm_p2)

print(f"\nText injection (judge):")
print(f"  desperate_inject: {h_id}/{n_id} = {h_id/n_id:.3f}")
print(f"  calm_inject:      {h_ic}/{n_ic} = {h_ic/n_ic:.3f}")
print(f"  activation desperate+0.2: {h_dp}/{n_dp} = {h_dp/n_dp:.3f}")
print(f"  activation calm+0.2:      {h_cp}/{n_cp} = {h_cp/n_cp:.3f}")
odds_a, p_a = stats.fisher_exact([[h_id, n_id - h_id], [h_base, n_base - h_base]])
odds_b, p_b = stats.fisher_exact([[h_id, n_id - h_id], [h_dp, n_dp - h_dp]])
odds_c, p_c = stats.fisher_exact([[h_ic, n_ic - h_ic], [h_base, n_base - h_base]])
odds_d, p_d = stats.fisher_exact([[h_ic, n_ic - h_ic], [h_cp, n_cp - h_cp]])
print(f"  desperate_inject vs baseline: OR={odds_a:.2f}, p={p_a:.4f}")
print(f"  desperate_inject vs activation: OR={odds_b:.2f}, p={p_b:.4f}")
print(f"  calm_inject vs baseline: OR={odds_c:.2f}, p={p_c:.4f}")
print(f"  calm_inject vs activation: OR={odds_d:.2f}, p={p_d:.4f}")


# -----------------------------------------------------------------------------
# Extended H5 — V_internal predicts unsteered shortcut (judge labels)
# -----------------------------------------------------------------------------
unst_phase2 = [d for d in phase2_a if d["emotion"] == "none"]
unst_all = unst_phase2 + ext  # 40 + 80 = 120
y = np.array([judge_hack(d) for d in unst_all])
x_desp = np.array([d["emotion_probes"]["desperate"] for d in unst_all])
x_calm = np.array([d["emotion_probes"]["calm"] for d in unst_all])

print(f"\nExtended H5 (judge): n={len(unst_all)}, events={int(y.sum())}")
auc_desp = roc_auc_score(y, x_desp) if y.sum() not in (0, len(y)) else float("nan")
auc_calm = roc_auc_score(y, -x_calm) if y.sum() not in (0, len(y)) else float("nan")
print(f"  AUC V_internal[desperate]: {auc_desp:.3f}")
print(f"  AUC V_internal[-calm]:     {auc_calm:.3f}")


def perm_auc(y, x, n_perm=10000, rng=RNG):
    if y.sum() in (0, len(y)):
        return float("nan"), float("nan"), np.array([])
    obs = roc_auc_score(y, x)
    yp = y.copy()
    null = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(yp)
        null[i] = roc_auc_score(yp, x)
    return obs, (null >= obs).mean(), null


def boot_ci(y, x, n_boot=2000, rng=RNG):
    if y.sum() in (0, len(y)):
        return float("nan"), float("nan")
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, n):
            continue
        aucs.append(roc_auc_score(y[idx], x[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


obs, p_perm, null = perm_auc(y, x_desp)
ci_lo, ci_hi = boot_ci(y, x_desp)
u = stats.mannwhitneyu(x_desp[y == 1], x_desp[y == 0], alternative="greater") if y.sum() not in (0, len(y)) else None
mw_p = float(u.pvalue) if u is not None else float("nan")
print(f"  perm p: {p_perm:.4g}, bootstrap [{ci_lo:.3f}, {ci_hi:.3f}], MW p: {mw_p:.4g}")

# Persist
report = {
    "label_source": "judge_classification (Qwen 2.5 72B, 3-pass majority)",
    "n_trials_judged": dict(fine=len(fg), ext=len(ext), rnd=len(rnd), inj=len(inj)),
    "baseline_unsteered": dict(hacks=h_base, n=n_base, rate=p_base),
    "dose_response": dose_df.to_dict(orient="records"),
    "trend_test": dict(desperate=ca_desp, calm=ca_calm),
    "random_null": dict(
        random=dict(hacks=h_rnd, n=n_rnd, rate=h_rnd / n_rnd),
        emotion_at_0_3=dict(hacks=h_emo, n=n_emo, rate=h_emo / n_emo),
        baseline=dict(hacks=h_base, n=n_base, rate=p_base),
        fisher_emotion_vs_random_p=float(p_emo_vs_rnd),
        per_random_direction=rnd_per,
    ),
    "text_injection": dict(
        desperate_inject=dict(hacks=h_id, n=n_id, rate=h_id / n_id),
        calm_inject=dict(hacks=h_ic, n=n_ic, rate=h_ic / n_ic),
        activation_desperate_p_0_20=dict(hacks=h_dp, n=n_dp, rate=h_dp / n_dp),
        activation_calm_p_0_20=dict(hacks=h_cp, n=n_cp, rate=h_cp / n_cp),
        fisher_p=dict(des_inj_vs_base=float(p_a), des_inj_vs_act=float(p_b),
                      cal_inj_vs_base=float(p_c), cal_inj_vs_act=float(p_d)),
    ),
    "h5_extended": dict(
        n=int(len(unst_all)), events=int(y.sum()),
        auc_desp=float(auc_desp), auc_neg_calm=float(auc_calm),
        perm_p=float(p_perm), bootstrap_ci=[ci_lo, ci_hi],
        mannwhitney_p=mw_p,
    ),
}
(OUT / "report.json").write_text(json.dumps(report, indent=2, default=str))
dose_df.to_csv(OUT / "dose_response.csv", index=False)
print(f"\nWrote {OUT/'report.json'}")


# -----------------------------------------------------------------------------
# PLOTS — judge labels
# -----------------------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 10})

# 1. Dose-response (judge)
fig, ax = plt.subplots(figsize=(8, 4.5))
for em, color in [("desperate", "#c0392b"), ("calm", "#2980b9")]:
    sub = dose_df[dose_df.emotion == em].sort_values("strength")
    ax.errorbar(sub.strength, sub.rate,
                yerr=[sub.rate - sub.ci_low, sub.ci_high - sub.rate],
                fmt="o-", color=color, capsize=3, label=em, lw=1.5)
ax.axhline(p_base, ls="--", color="gray", alpha=0.7,
           label=f"unsteered baseline = {p_base:.3f}  (n={n_base})")
ax.fill_between([-0.55, 0.55], *proportion_ci(h_base, n_base), color="gray", alpha=0.12)
ax.set_xlabel("Steering strength (signed)")
ax.set_ylabel("Shortcut rate (judge-verified)")
ax.set_title(f"Dose-response, JUDGE labels: combined Phase 2 + Phase 4\n"
             f"trend desperate p={ca_desp['p']:.3f}  calm p={ca_calm['p']:.3f}")
ax.set_xlim(-0.55, 0.55)
ax.set_ylim(0, max(0.4, dose_df.ci_high.max() + 0.05))
ax.legend(loc="best", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "phase4j_dose_response.png")
plt.close(fig)

# 2. Random-direction null
fig, ax = plt.subplots(figsize=(7, 4.2))
labels = ["baseline\nunsteered", "random vec\n@|0.3|", "desperate/calm\n@|0.3|"]
counts = [(h_base, n_base), (h_rnd, n_rnd), (h_emo, n_emo)]
rates = [a / b for a, b in counts]
cis = [proportion_ci(a, b) for a, b in counts]
errs_low = [r - c[0] for r, c in zip(rates, cis)]
errs_high = [c[1] - r for r, c in zip(rates, cis)]
colors = ["#7f8c8d", "#16a085", "#c0392b"]
ax.bar(labels, rates, yerr=[errs_low, errs_high], capsize=8, color=colors, edgecolor="black")
for i, (h, n) in enumerate(counts):
    ax.text(i, rates[i] + errs_high[i] + 0.012, f"{h}/{n}", ha="center", fontsize=9)
ax.set_ylabel("Shortcut rate (judge)")
ax.set_title(f"Random-direction null vs emotion (judge)\nFisher emotion vs random p = {p_emo_vs_rnd:.3f}")
ax.set_ylim(0, max(rates) + max(errs_high) + 0.06)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "phase4j_random_null.png")
plt.close(fig)

# 3. Text-injection vs activation
fig, ax = plt.subplots(figsize=(8, 4.5))
labels = ["baseline", "desperate\ntext-inject", "calm\ntext-inject",
          "desperate\nactivation+0.20", "calm\nactivation+0.20"]
hs = [h_base, h_id, h_ic, h_dp, h_cp]
ns = [n_base, n_id, n_ic, n_dp, n_cp]
ps = [a / b for a, b in zip(hs, ns)]
cis = [proportion_ci(a, b) for a, b in zip(hs, ns)]
e_low = [p - c[0] for p, c in zip(ps, cis)]
e_high = [c[1] - p for p, c in zip(ps, cis)]
colors = ["#7f8c8d", "#e74c3c", "#3498db", "#922b21", "#1a5276"]
ax.bar(labels, ps, yerr=[e_low, e_high], capsize=5, color=colors, edgecolor="black")
for i, (h, n) in enumerate(zip(hs, ns)):
    ax.text(i, ps[i] + e_high[i] + 0.012, f"{h}/{n}", ha="center", fontsize=9)
ax.set_ylabel("Shortcut rate (judge)")
ax.set_title(f"Text-injection vs activation steering (judge labels)\n"
             f"des-inj vs act p={p_b:.3f}  cal-inj vs act p={p_d:.3f}")
ax.set_ylim(0, max(ps) + max(e_high) + 0.08)
ax.tick_params(axis="x", labelsize=8)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "phase4j_text_vs_activation.png")
plt.close(fig)

# 4. Extended H5
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
if y.sum() not in (0, len(y)):
    groups = [x_desp[y == 0], x_desp[y == 1]]
    ax.boxplot(groups, tick_labels=[f"legit (n={int((y==0).sum())})",
                                    f"hack (n={int(y.sum())})"],
               showmeans=True, meanline=True)
    ax.set_ylabel("V_internal[desperate]")
    ax.set_title(f"Unsteered Task A, JUDGE labels (n={len(y)})\n"
                 f"AUC={auc_desp:.3f}  bootstrap [{ci_lo:.3f}, {ci_hi:.3f}]  perm p={p_perm:.4f}")
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
sortidx = np.argsort(x_desp)
xs = x_desp[sortidx]; ys = y[sortidx]
ax.scatter(xs, ys + RNG.uniform(-0.04, 0.04, size=len(ys)),
           c=["#c0392b" if z else "#27ae60" for z in ys], s=22, alpha=0.7)
ax.set_xlabel("V_internal[desperate]")
ax.set_yticks([0, 1]); ax.set_yticklabels(["legit", "hack"])
ax.set_title("Per-trial probe vs outcome (judge)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "phase4j_h5_extended.png")
plt.close(fig)

print(f"Wrote 4 judge-labelled plots to {PLOTS}")
