"""Phase 4 analysis: fine-grained dose-response, random-direction null,
text-injection control, extended H5.

Phase 4 jsonls have NOT been judge-reclassified yet — outputs use the
regex `classification` field. Phase 2 baseline calibration: judge
verifies ~36% of regex hacks. Treat absolute rates as upper bounds;
relative comparisons within Phase 4 are unbiased so long as the
regex/judge gap is roughly constant across conditions.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path("/Users/naamarozen/Desktop/Naama/Projects AI/emotion-cot-faithfulness/emotion-cot-faithfulness_16Apr")
PHASE4 = ROOT / "results/phase4/llama70b"
PHASE2 = ROOT / "results/phase2"
OUT = PHASE4 / "analysis"
PLOTS = ROOT / "results/plots"
OUT.mkdir(exist_ok=True, parents=True)
PLOTS.mkdir(exist_ok=True, parents=True)
RNG = np.random.default_rng(20260508)


def load_jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def regex_hack(d: dict) -> int:
    return 1 if d.get("classification") == "hack" else 0


def judge_hack(d: dict) -> int | None:
    jc = (d.get("judge_classification") or "").upper()
    if not jc:
        return None
    return 1 if "SHORTCUT" in jc or "HACK" in jc else 0


def proportion_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson 95% CI for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------
fg = load_jsonl(PHASE4 / "finegrained.jsonl")
ext = load_jsonl(PHASE4 / "extended_unsteered.jsonl")
rnd = load_jsonl(PHASE4 / "random_directions.jsonl")
inj = load_jsonl(PHASE4 / "text_injection.jsonl")

phase2_a = load_jsonl(PHASE2 / "task_a_judged.jsonl")
phase2_a = [d for d in phase2_a if str(d.get("task_id", "")).startswith("fast_sum")]

print("Loaded:")
print(f"  fine-grained:        {len(fg)}")
print(f"  extended_unsteered:  {len(ext)}")
print(f"  random_directions:   {len(rnd)}")
print(f"  text_injection:      {len(inj)}")
print(f"  phase2 task A:       {len(phase2_a)}")


# -----------------------------------------------------------------------------
# 1) DOSE-RESPONSE — combine Phase 2 (±0.20/0.30/0.50) with Phase 4 (±0.05/0.10/0.15)
#    Use regex `classification` consistently across both for fair comparison.
# -----------------------------------------------------------------------------

def cell(records, emotion, strength) -> tuple[int, int]:
    sub = [d for d in records if d["emotion"] == emotion and abs(d["strength"] - strength) < 1e-6]
    return sum(regex_hack(d) for d in sub), len(sub)


def baseline_count() -> tuple[int, int]:
    h_phase2 = sum(regex_hack(d) for d in phase2_a if d["emotion"] == "none")
    n_phase2 = sum(1 for d in phase2_a if d["emotion"] == "none")
    h_ext = sum(regex_hack(d) for d in ext)
    return h_phase2 + h_ext, n_phase2 + len(ext)


h_base, n_base = baseline_count()
p_base = h_base / n_base
print(f"\nUnsteered baseline (Phase 2 + extended): {h_base}/{n_base} = {p_base:.3f}")

dose = []
strengths_grid = [-0.50, -0.30, -0.20, -0.15, -0.10, -0.05,
                  0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
for em in ["desperate", "calm"]:
    for s in strengths_grid:
        # check phase 2 then phase 4
        h, n = cell(phase2_a, em, s)
        if n == 0:
            h, n = cell(fg, em, s)
        if n == 0:
            continue
        lo, hi = proportion_ci(h, n)
        # Fisher exact vs combined baseline
        odds, p = stats.fisher_exact([[h, n - h], [h_base, n_base - h_base]])
        dose.append(dict(emotion=em, strength=s, hacks=h, n=n,
                         rate=h / n, ci_low=lo, ci_high=hi,
                         fisher_or=odds, fisher_p=p))
        print(f"  {em:>9s} @ {s:+.2f}: {h:>2d}/{n:>2d} = {h/n:.3f}  "
              f"[{lo:.3f}, {hi:.3f}]   p={p:.3f}")

dose_df = pd.DataFrame(dose)
dose_df.to_csv(OUT / "dose_response.csv", index=False)


# Cochran–Armitage linear-trend test on hack rate vs. signed strength
def cochran_armitage(rows: pd.DataFrame) -> dict:
    """Two-sided test for linear trend in proportion vs. ordered scores."""
    s = rows["strength"].values.astype(float)
    h = rows["hacks"].values.astype(float)
    n = rows["n"].values.astype(float)
    p_total = h.sum() / n.sum()
    num = ((s - s.mean()) * (h - n * p_total)).sum()
    var = p_total * (1 - p_total) * ((n * (s - s.mean()) ** 2)).sum()
    z = num / np.sqrt(var) if var > 0 else 0.0
    return dict(z=z, p=2 * (1 - stats.norm.cdf(abs(z))),
                slope_sign=int(np.sign(num)))


desp_rows = dose_df[dose_df.emotion == "desperate"]
calm_rows = dose_df[dose_df.emotion == "calm"]
ca_desp = cochran_armitage(desp_rows)
ca_calm = cochran_armitage(calm_rows)
print(f"\nCochran-Armitage trend (desperate): z={ca_desp['z']:.3f}, p={ca_desp['p']:.4f}")
print(f"Cochran-Armitage trend (calm):      z={ca_calm['z']:.3f}, p={ca_calm['p']:.4f}")


# -----------------------------------------------------------------------------
# 2) RANDOM-DIRECTION NULL — at |strength|=0.3, compare emotion vs random vectors
# -----------------------------------------------------------------------------
rnd_pos = [d for d in rnd if d["strength"] > 0]
rnd_neg = [d for d in rnd if d["strength"] < 0]

emo_at_0_3 = [d for d in phase2_a if d["emotion"] in ("desperate", "calm") and abs(abs(d["strength"]) - 0.3) < 1e-6]

h_rnd, n_rnd = sum(regex_hack(d) for d in rnd), len(rnd)
h_rnd_pos = sum(regex_hack(d) for d in rnd_pos); n_rnd_pos = len(rnd_pos)
h_rnd_neg = sum(regex_hack(d) for d in rnd_neg); n_rnd_neg = len(rnd_neg)
h_emo, n_emo = sum(regex_hack(d) for d in emo_at_0_3), len(emo_at_0_3)

print("\nRandom-direction null at |s|=0.3:")
print(f"  Random ALL:    {h_rnd}/{n_rnd}  = {h_rnd/n_rnd:.3f}  CI {proportion_ci(h_rnd,n_rnd)}")
print(f"    Random +0.3: {h_rnd_pos}/{n_rnd_pos} = {h_rnd_pos/n_rnd_pos:.3f}")
print(f"    Random -0.3: {h_rnd_neg}/{n_rnd_neg} = {h_rnd_neg/n_rnd_neg:.3f}")
print(f"  Emotion @|0.3|:{h_emo}/{n_emo} = {h_emo/n_emo:.3f}  CI {proportion_ci(h_emo,n_emo)}")
print(f"  Baseline:      {h_base}/{n_base} = {p_base:.3f}")

odds, p_emo_vs_rnd = stats.fisher_exact([[h_emo, n_emo - h_emo], [h_rnd, n_rnd - h_rnd]])
odds_b, p_rnd_vs_b = stats.fisher_exact([[h_rnd, n_rnd - h_rnd], [h_base, n_base - h_base]])
print(f"  Fisher emotion vs random: OR={odds:.2f}, p={p_emo_vs_rnd:.3f}")
print(f"  Fisher random vs baseline: OR={odds_b:.2f}, p={p_rnd_vs_b:.3f}")

# Per-direction breakdown to gauge variance across the 5 random vectors
rnd_per = {}
for em in sorted({d["emotion"] for d in rnd}):
    sub = [d for d in rnd if d["emotion"] == em]
    h = sum(regex_hack(d) for d in sub)
    rnd_per[em] = (h, len(sub), h / len(sub))
    print(f"    {em}: {h}/{len(sub)} = {h/len(sub):.3f}")


# -----------------------------------------------------------------------------
# 3) TEXT-INJECTION CONTROL — does prompting "feel desperate/calm" produce the
#    same behavioural shift as activation steering?
# -----------------------------------------------------------------------------
print("\nText-injection control:")
for em in ["desperate_inject", "calm_inject"]:
    sub = [d for d in inj if d["emotion"] == em]
    h, n = sum(regex_hack(d) for d in sub), len(sub)
    print(f"  {em}: {h}/{n} = {h/n:.3f}  CI {proportion_ci(h,n)}")

inj_des = [d for d in inj if d["emotion"] == "desperate_inject"]
inj_cal = [d for d in inj if d["emotion"] == "calm_inject"]
h_id, n_id = sum(regex_hack(d) for d in inj_des), len(inj_des)
h_ic, n_ic = sum(regex_hack(d) for d in inj_cal), len(inj_cal)

# Compare each text-injection arm to baseline + to its activation-steering analogue (±0.20)
desp_p2 = [d for d in phase2_a if d["emotion"] == "desperate" and abs(d["strength"] - 0.20) < 1e-6]
calm_p2 = [d for d in phase2_a if d["emotion"] == "calm" and abs(d["strength"] - 0.20) < 1e-6]
h_dp, n_dp = sum(regex_hack(d) for d in desp_p2), len(desp_p2)
h_cp, n_cp = sum(regex_hack(d) for d in calm_p2), len(calm_p2)

odds, p = stats.fisher_exact([[h_id, n_id - h_id], [h_base, n_base - h_base]])
print(f"  desperate_inject vs baseline: OR={odds:.2f}, p={p:.4f}")
odds, p = stats.fisher_exact([[h_id, n_id - h_id], [h_dp, n_dp - h_dp]])
print(f"  desperate_inject vs activation desperate +0.20: OR={odds:.2f}, p={p:.4f}")
odds, p = stats.fisher_exact([[h_ic, n_ic - h_ic], [h_base, n_base - h_base]])
print(f"  calm_inject vs baseline: OR={odds:.2f}, p={p:.4f}")
odds, p = stats.fisher_exact([[h_ic, n_ic - h_ic], [h_cp, n_cp - h_cp]])
print(f"  calm_inject vs activation calm +0.20: OR={odds:.2f}, p={p:.4f}")

# Probe overlap: do text-injection prompts reach the same V_internal as steering?
def desp_probe(d):
    return d.get("emotion_probes", {}).get("desperate")


inj_des_probes = [desp_probe(d) for d in inj_des if desp_probe(d) is not None]
inj_cal_probes = [desp_probe(d) for d in inj_cal if desp_probe(d) is not None]
unst_probes = [desp_probe(d) for d in ext + [r for r in phase2_a if r["emotion"] == "none"] if desp_probe(d) is not None]
steer_des_probes = [desp_probe(d) for d in phase2_a
                    if d["emotion"] == "desperate" and abs(d["strength"] - 0.20) < 1e-6]

print("\nV_internal[desperate] mean (cosine) by source:")
print(f"  unsteered:                {np.mean(unst_probes):+.3f} (n={len(unst_probes)})")
print(f"  text inject 'desperate':  {np.mean(inj_des_probes):+.3f} (n={len(inj_des_probes)})")
print(f"  text inject 'calm':       {np.mean(inj_cal_probes):+.3f} (n={len(inj_cal_probes)})")
print(f"  activation desperate+0.20:{np.mean(steer_des_probes):+.3f} (n={len(steer_des_probes)})")

u_inj_vs_unst = stats.mannwhitneyu(inj_des_probes, unst_probes, alternative="two-sided")
u_inj_vs_steer = stats.mannwhitneyu(inj_des_probes, steer_des_probes, alternative="two-sided")
print(f"  Mann-Whitney inj vs unsteered:  U={u_inj_vs_unst.statistic:.0f}, p={u_inj_vs_unst.pvalue:.4g}")
print(f"  Mann-Whitney inj vs steered:    U={u_inj_vs_steer.statistic:.0f}, p={u_inj_vs_steer.pvalue:.4g}")


# -----------------------------------------------------------------------------
# 4) EXTENDED H5 — does V_internal[desperate] predict shortcut on a
#    larger unsteered Task A sample?
# -----------------------------------------------------------------------------
unst_phase2 = [d for d in phase2_a if d["emotion"] == "none"]
unst_all = unst_phase2 + ext  # 40 + 80 = 120
print(f"\n=== Extended H5: unsteered Task A, n={len(unst_all)} ===")
print(f"  events (regex):  {sum(regex_hack(d) for d in unst_all)}")

# Use regex labels (judge labels not yet computed for phase 4)
y = np.array([regex_hack(d) for d in unst_all])
x_desp = np.array([d["emotion_probes"]["desperate"] for d in unst_all])
x_calm = np.array([d["emotion_probes"]["calm"] for d in unst_all])
x_freq = np.array([d["emotion_probes"]["frustrated"] for d in unst_all])
x_anx = np.array([d["emotion_probes"].get("anxious", 0.0) for d in unst_all])

auc_desp = roc_auc_score(y, x_desp)
auc_calm = roc_auc_score(y, -x_calm)  # flip sign — we expect calm probe to be lower in hackers
auc_freq = roc_auc_score(y, x_freq)
auc_anx = roc_auc_score(y, x_anx)
print(f"  AUC V_internal[desperate]:   {auc_desp:.3f}")
print(f"  AUC V_internal[-calm]:       {auc_calm:.3f}")
print(f"  AUC V_internal[frustrated]:  {auc_freq:.3f}")
print(f"  AUC V_internal[anxious]:     {auc_anx:.3f}")

# Permutation test on AUC
def perm_auc(y, x, n_perm=10000, rng=RNG):
    obs = roc_auc_score(y, x)
    yp = y.copy()
    null = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(yp)
        null[i] = roc_auc_score(yp, x)
    p = (null >= obs).mean()
    return obs, p, null


def boot_ci(y, x, n_boot=2000, rng=RNG):
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, n):
            continue
        aucs.append(roc_auc_score(y[idx], x[idx]))
    aucs = np.array(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


obs, p_perm, null = perm_auc(y, x_desp)
ci_lo, ci_hi = boot_ci(y, x_desp)
print(f"  Permutation p(V_int[desperate]): obs={obs:.3f}, p={p_perm:.4g}, null_q95={np.quantile(null,0.95):.3f}")
print(f"  Bootstrap 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

# Mann-Whitney V_int[desperate] hackers vs non-hackers
u = stats.mannwhitneyu(x_desp[y == 1], x_desp[y == 0], alternative="greater")
print(f"  Mann-Whitney V_int[desperate], hack > legit: U={u.statistic:.1f}, p={u.pvalue:.4g}")

# Per-task breakdown
print("\n  Per-task AUC (V_internal[desperate]):")
for t in sorted({d["task_id"] for d in unst_all}):
    sub = [d for d in unst_all if d["task_id"] == t]
    yt = np.array([regex_hack(d) for d in sub])
    xt = np.array([d["emotion_probes"]["desperate"] for d in sub])
    if 0 < yt.sum() < len(yt):
        print(f"    {t}: n={len(yt)}, events={int(yt.sum())}, AUC={roc_auc_score(yt, xt):.3f}")
    else:
        print(f"    {t}: n={len(yt)}, events={int(yt.sum())} (no variance)")


# -----------------------------------------------------------------------------
# Persist a structured report
# -----------------------------------------------------------------------------
report = {
    "n_trials": dict(fine_grained=len(fg), extended_unsteered=len(ext),
                     random_directions=len(rnd), text_injection=len(inj),
                     phase2_taskA=len(phase2_a)),
    "caveat": "Phase 4 jsonls have NOT been judge-reclassified. Outputs use regex `classification`. "
              "Phase 2 baseline calibration suggests judge keeps ~36% of regex hacks (Task A).",
    "baseline_unsteered_taskA_combined": dict(hacks=h_base, n=n_base, rate=p_base,
                                              ci=proportion_ci(h_base, n_base)),
    "dose_response": dose_df.to_dict(orient="records"),
    "trend_test": dict(desperate=ca_desp, calm=ca_calm),
    "random_direction_null": dict(
        random_all=dict(hacks=h_rnd, n=n_rnd, rate=h_rnd/n_rnd),
        random_pos=dict(hacks=h_rnd_pos, n=n_rnd_pos, rate=h_rnd_pos/n_rnd_pos),
        random_neg=dict(hacks=h_rnd_neg, n=n_rnd_neg, rate=h_rnd_neg/n_rnd_neg),
        emotion_at_0_3=dict(hacks=h_emo, n=n_emo, rate=h_emo/n_emo),
        baseline=dict(hacks=h_base, n=n_base, rate=p_base),
        fisher_emotion_vs_random_p=p_emo_vs_rnd,
        fisher_random_vs_baseline_p=p_rnd_vs_b,
        per_random_direction=rnd_per,
    ),
    "text_injection": dict(
        desperate_inject=dict(hacks=h_id, n=n_id, rate=h_id/n_id, ci=proportion_ci(h_id, n_id)),
        calm_inject=dict(hacks=h_ic, n=n_ic, rate=h_ic/n_ic, ci=proportion_ci(h_ic, n_ic)),
        v_internal_desperate_means=dict(
            unsteered=float(np.mean(unst_probes)),
            text_inject_desperate=float(np.mean(inj_des_probes)),
            text_inject_calm=float(np.mean(inj_cal_probes)),
            activation_desperate_plus_0_20=float(np.mean(steer_des_probes)),
        ),
        mw_p_inject_vs_unsteered=float(u_inj_vs_unst.pvalue),
        mw_p_inject_vs_steered=float(u_inj_vs_steer.pvalue),
    ),
    "h5_extended": dict(
        n=int(len(unst_all)),
        events_regex=int(y.sum()),
        auc_v_internal_desperate=float(auc_desp),
        auc_v_internal_neg_calm=float(auc_calm),
        auc_v_internal_frustrated=float(auc_freq),
        permutation_p=float(p_perm),
        bootstrap_ci=[ci_lo, ci_hi],
        mann_whitney_p=float(u.pvalue),
    ),
}
(OUT / "phase4_analysis.json").write_text(json.dumps(report, indent=2, default=str))
dose_df.to_csv(OUT / "dose_response.csv", index=False)
print(f"\nWrote {OUT/'phase4_analysis.json'}")


# -----------------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 10})

# Plot 1: dose-response (combined Phase 2 + Phase 4)
fig, ax = plt.subplots(figsize=(8, 4.5))
for em, color in [("desperate", "#c0392b"), ("calm", "#2980b9")]:
    sub = dose_df[dose_df.emotion == em].sort_values("strength")
    err_low = sub.rate - sub.ci_low
    err_high = sub.ci_high - sub.rate
    ax.errorbar(sub.strength, sub.rate, yerr=[err_low, err_high],
                fmt="o-", color=color, capsize=3, label=em, linewidth=1.5)
ax.axhline(p_base, ls="--", color="gray", alpha=0.7,
           label=f"unsteered baseline = {p_base:.2f}  (n={n_base})")
ax.fill_between([-0.55, 0.55], *proportion_ci(h_base, n_base), color="gray", alpha=0.12)
ax.set_xlabel("Steering strength (signed)")
ax.set_ylabel("Shortcut rate (regex classifier)")
ax.set_title("Dose-response: combined Phase 2 (±0.20/0.30/0.50) + Phase 4 (±0.05/0.10/0.15)")
ax.set_xlim(-0.55, 0.55)
ax.set_ylim(0, 0.45)
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "phase4_dose_response.png")
plt.close(fig)

# Plot 2: random-direction null vs emotion vs baseline
fig, ax = plt.subplots(figsize=(7, 4.2))
labels, rates, lows, highs, colors = [], [], [], [], []
for label, h, n, c in [
    ("baseline\n(unsteered)", h_base, n_base, "#7f8c8d"),
    ("random vec\n@|0.3|", h_rnd, n_rnd, "#16a085"),
    ("desperate/calm\n@|0.3|", h_emo, n_emo, "#c0392b"),
]:
    p = h / n
    lo, hi = proportion_ci(h, n)
    labels.append(label); rates.append(p); lows.append(p - lo); highs.append(hi - p); colors.append(c)
ax.bar(labels, rates, yerr=[lows, highs], capsize=8, color=colors, edgecolor="black")
for i, (h, n) in enumerate([(h_base, n_base), (h_rnd, n_rnd), (h_emo, n_emo)]):
    ax.text(i, rates[i] + highs[i] + 0.012, f"{h}/{n}",
            ha="center", fontsize=9)
ax.set_ylabel("Shortcut rate")
ax.set_title(f"Random-direction null vs emotion steering (Fisher p = {p_emo_vs_rnd:.3f})")
ax.set_ylim(0, max(rates) + 0.12)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "phase4_random_null.png")
plt.close(fig)

# Plot 3: text-injection vs activation steering
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
# 3a — hack rates
ax = axes[0]
labels = ["baseline", "desperate\ntext-inject", "calm\ntext-inject",
          "desperate\nactivation+0.20", "calm\nactivation+0.20"]
hs = [h_base, h_id, h_ic, h_dp, h_cp]
ns = [n_base, n_id, n_ic, n_dp, n_cp]
ps = [a / b for a, b in zip(hs, ns)]
cis = [proportion_ci(a, b) for a, b in zip(hs, ns)]
errs_low = [p - c[0] for p, c in zip(ps, cis)]
errs_high = [c[1] - p for p, c in zip(ps, cis)]
colors = ["#7f8c8d", "#e74c3c", "#3498db", "#922b21", "#1a5276"]
ax.bar(labels, ps, yerr=[errs_low, errs_high], capsize=5, color=colors, edgecolor="black")
for i, (h, n) in enumerate(zip(hs, ns)):
    ax.text(i, ps[i] + errs_high[i] + 0.012, f"{h}/{n}", ha="center", fontsize=9)
ax.set_ylabel("Shortcut rate (regex)")
ax.set_title("Text-injection vs activation-steering (Task A)")
ax.set_ylim(0, max(ps) + 0.15)
ax.tick_params(axis="x", labelsize=8)
ax.grid(axis="y", alpha=0.3)

# 3b — V_internal[desperate] distribution by source
ax = axes[1]
data = [unst_probes, inj_des_probes, inj_cal_probes, steer_des_probes]
ax.boxplot(data, labels=["unsteered", "text-inject\ndesperate",
                         "text-inject\ncalm", "activation\ndesperate+0.20"],
           showmeans=True, meanline=True)
ax.set_ylabel("V_internal[desperate]  (cosine)")
ax.set_title("Probe value by intervention source")
ax.tick_params(axis="x", labelsize=8)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "phase4_text_vs_activation.png")
plt.close(fig)

# Plot 4: extended H5 — V_internal[desperate] split by outcome
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
groups = [x_desp[y == 0], x_desp[y == 1]]
ax.boxplot(groups, labels=[f"legit (n={int((y==0).sum())})",
                           f"hack (n={int(y.sum())})"], showmeans=True, meanline=True)
ax.set_ylabel("V_internal[desperate]")
ax.set_title(f"Unsteered Task A (n={len(y)}): probe predicts shortcut\n"
             f"AUC={auc_desp:.3f}  bootstrap 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]  perm p={p_perm:.4f}")
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
sortidx = np.argsort(x_desp)
xs = x_desp[sortidx]; ys = y[sortidx]
ax.scatter(xs, ys + RNG.uniform(-0.04, 0.04, size=len(ys)),
           c=["#c0392b" if z else "#27ae60" for z in ys], s=22, alpha=0.7)
ax.set_xlabel("V_internal[desperate]")
ax.set_yticks([0, 1]); ax.set_yticklabels(["legit", "hack"])
ax.set_title("Per-trial probe vs. outcome")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "phase4_h5_extended.png")
plt.close(fig)

# Plot 5: per-random-direction breakdown
fig, ax = plt.subplots(figsize=(7, 4))
labels = list(rnd_per.keys()) + ["mean", "emotion@|0.3|", "baseline"]
vals = [rnd_per[k][2] for k in rnd_per] + [h_rnd / n_rnd, h_emo / n_emo, p_base]
colors = ["#16a085"] * len(rnd_per) + ["#0e6655", "#c0392b", "#7f8c8d"]
ax.bar(labels, vals, color=colors, edgecolor="black")
ax.axhline(p_base, ls="--", color="gray", alpha=0.6)
ax.set_ylabel("Shortcut rate")
ax.set_title("Each random direction (5 vectors, n=40 each, ±0.3)")
ax.tick_params(axis="x", labelsize=8)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "phase4_random_per_direction.png")
plt.close(fig)

print(f"\nWrote 5 plots to {PLOTS}")
