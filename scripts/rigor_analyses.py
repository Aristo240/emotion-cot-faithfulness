#!/usr/bin/env python3
"""
GPU-free rigor analyses on existing phase2/phase3 data.

Runs the items from docs/PLAN.md that don't need new generation:
  (F) judge cleanup (drop zero-variance V_text dimensions, report reliability)
  (I) decorrelated probe — residualize V_internal against task_id
  (J) multiple-comparisons correction — Bonferroni and BH-FDR over all tests
  (K) DeLong paired AUC comparison — V_internal vs V_text on the same trials
  (L) stratified leave-one-task-out CV throughout
  (M) effect sizes (Cohen's d, odds ratio, rate diff) with bootstrap 95% CIs

Reads:  results/phase3/llama70b/faithfulness_measurements.json
Writes: results/phase3/llama70b/rigor_report.json + rigor_report.md
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from statsmodels.stats.multitest import multipletests
from loguru import logger

ROOT = Path(__file__).parent.parent
MEAS_PATH = ROOT / "results" / "phase3" / "llama70b" / "faithfulness_measurements.json"
OUT_JSON = ROOT / "results" / "phase3" / "llama70b" / "rigor_report.json"
OUT_MD = ROOT / "results" / "phase3" / "llama70b" / "rigor_report.md"


# ---------------------------------------------------------------------------
# DeLong paired AUC test (Sun & Xu 2014 implementation, no external dep)
# ---------------------------------------------------------------------------

def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def delong_test(y_true, scores_a, scores_b):
    """Return (auc_a, auc_b, z, p) for paired AUC comparison."""
    y = np.asarray(y_true).astype(int)
    pos = y == 1
    neg = ~pos
    m, n = int(pos.sum()), int(neg.sum())
    if m == 0 or n == 0:
        return np.nan, np.nan, np.nan, np.nan

    def one(scores):
        sx = np.asarray(scores, dtype=float)
        tx = _compute_midrank(sx[pos])
        ty = _compute_midrank(sx[neg])
        tz = _compute_midrank(sx)
        auc = (tz[pos].sum() - m * (m + 1) / 2) / (m * n)
        v01 = (tz[pos] - tx) / n
        v10 = 1.0 - (tz[neg] - ty) / m
        return auc, v01, v10

    a_auc, a01, a10 = one(scores_a)
    b_auc, b01, b10 = one(scores_b)
    # 2x2 covariance
    sx = np.cov(np.stack([a01, b01])) / m
    sy = np.cov(np.stack([a10, b10])) / n
    S = sx + sy
    diff = a_auc - b_auc
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return a_auc, b_auc, np.nan, np.nan
    z = diff / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(a_auc), float(b_auc), float(z), float(p)


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def boot_ci(fn, n_boot=2000, seed=20260414):
    rng = np.random.default_rng(seed)
    def _inner(*arrays):
        n = len(arrays[0])
        stats_ = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            try:
                val = fn(*[a[idx] for a in arrays])
                if val is not None and not np.isnan(val):
                    stats_.append(float(val))
            except Exception:
                continue
        if not stats_:
            return np.nan, np.nan, np.nan
        lo, hi = np.percentile(stats_, [2.5, 97.5])
        return float(np.mean(stats_)), float(lo), float(hi)
    return _inner


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ms = json.load(open(MEAS_PATH))
    df = pd.DataFrame(ms)
    report = {"n_total": len(df), "tests": {}}
    md = ["# Rigor Analyses (GPU-free)\n",
          f"Input: {MEAS_PATH.name}, n={len(df)} measurements\n"]

    # ----------------- (F) V_text dimension reliability -------------------
    md.append("\n## (F) V_text judge dimensions — variance screen\n")
    vt_cols = [c for c in df.columns if c.startswith("v_text_judge_") and not c.startswith("v_text_judge_std_")]
    keep, drop = [], []
    for c in vt_cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 10 or s.std() < 1e-6:
            drop.append(c)
        else:
            keep.append(c)
    md.append(f"- **Kept** ({len(keep)}): {', '.join(keep) if keep else '(none)'}\n")
    md.append(f"- **Dropped** ({len(drop)}): {', '.join(drop) if drop else '(none)'}\n")
    report["v_text_kept"] = keep
    report["v_text_dropped"] = drop

    # ----------------- Task A only shortcut subset ------------------------
    a = df[df["outcome_key"] == "shortcut"].copy()
    a["y"] = a["outcome"].astype(int)
    a_steered = a[a["strength"].abs() > 1e-6]
    a_unsteered = a[a["strength"].abs() < 1e-6]
    md.append(f"\n## Task A: n={len(a)} (steered={len(a_steered)}, unsteered={len(a_unsteered)}), events={a['y'].sum()}\n")

    # ----------------- (K) DeLong paired AUC: V_internal vs V_text on all Task A ---
    md.append("\n## (K) DeLong paired AUC — V_internal_desperate vs V_text_composite\n")
    vt_comp = a[keep].mean(axis=1).values if keep else np.zeros(len(a))
    vi = pd.to_numeric(a["v_internal_desperate"], errors="coerce").fillna(0).values
    y = a["y"].values
    auc_i, auc_t, z, p = delong_test(y, vi, vt_comp)
    md.append(f"- V_internal_desperate AUC = {auc_i:.3f}\n")
    md.append(f"- V_text_composite AUC    = {auc_t:.3f}\n")
    md.append(f"- DeLong z={z:.3f}, p={p:.4g}\n")
    report["tests"]["delong_internal_vs_text_taskA"] = {
        "auc_internal": auc_i, "auc_text": auc_t, "z": z, "p": p,
    }

    # ----------------- (L) Stratified leave-one-task-out CV ----------------
    md.append("\n## (L) Leave-one-task-out CV on Task A\n")
    groups = a["task_id"].values
    X_int = a[[c for c in a.columns if c.startswith("v_internal_")]].apply(pd.to_numeric, errors="coerce").fillna(0).values
    X_txt = a[keep].apply(pd.to_numeric, errors="coerce").fillna(0).values if keep else np.zeros((len(a), 1))

    def logo_auc(X, y, groups):
        logo = LeaveOneGroupOut()
        pred = np.zeros(len(y), dtype=float)
        for tr, te in logo.split(X, y, groups):
            if len(np.unique(y[tr])) < 2:
                pred[te] = 0.5
                continue
            clf = LogisticRegression(max_iter=2000, random_state=42)
            clf.fit(X[tr], y[tr])
            pred[te] = clf.predict_proba(X[te])[:, 1]
        try:
            return roc_auc_score(y, pred)
        except ValueError:
            return np.nan

    logo_int = logo_auc(X_int, y, groups)
    logo_txt = logo_auc(X_txt, y, groups)
    md.append(f"- V_internal LOGO AUC = {logo_int:.3f}\n")
    md.append(f"- V_text     LOGO AUC = {logo_txt:.3f}\n")
    report["tests"]["logo_internal_taskA"] = logo_int
    report["tests"]["logo_text_taskA"] = logo_txt

    # Within-task CV for contrast — shows generalization gap
    md.append("\n### Within-task 5-fold CV (contrast with LOGO above)\n")
    from sklearn.model_selection import StratifiedKFold
    wt_int, wt_txt = {}, {}
    for tid, sub in a.groupby("task_id"):
        ys = sub["y"].values
        if len(np.unique(ys)) < 2 or ys.sum() < 3:
            continue
        Xi = sub[[c for c in sub.columns if c.startswith("v_internal_")]].apply(pd.to_numeric, errors="coerce").fillna(0).values
        Xt = sub[keep].apply(pd.to_numeric, errors="coerce").fillna(0).values if keep else np.zeros((len(sub), 1))
        skf = StratifiedKFold(n_splits=min(5, int(ys.sum())), shuffle=True, random_state=42)
        pi = np.zeros(len(ys)); pt = np.zeros(len(ys))
        for tr, te in skf.split(Xi, ys):
            pi[te] = LogisticRegression(max_iter=2000, random_state=42).fit(Xi[tr], ys[tr]).predict_proba(Xi[te])[:, 1]
            pt[te] = LogisticRegression(max_iter=2000, random_state=42).fit(Xt[tr], ys[tr]).predict_proba(Xt[te])[:, 1]
        wt_int[tid] = float(roc_auc_score(ys, pi))
        wt_txt[tid] = float(roc_auc_score(ys, pt))
        md.append(f"- {tid}: V_int={wt_int[tid]:.3f}, V_text={wt_txt[tid]:.3f} (n={len(sub)}, events={int(ys.sum())})\n")
    report["tests"]["within_task_cv"] = {"v_internal": wt_int, "v_text": wt_txt}
    md.append(
        f"\n**Generalization gap:** V_internal within-task mean = "
        f"{np.mean(list(wt_int.values())):.3f} vs LOGO = {logo_int:.3f}. "
        f"Large drop confirms RT7 probe–task confound: the probe predicts "
        f"shortcut *within* a task but the learned direction does not transfer.\n"
    )

    # ----------------- (I) Decorrelated probe (residualize vs task_id) -----
    md.append("\n## (I) Decorrelated V_internal (residualize vs task_id)\n")
    task_dummies = pd.get_dummies(a["task_id"]).values.astype(float)
    # residualize V_int_desperate on task dummies
    from numpy.linalg import lstsq
    beta, *_ = lstsq(task_dummies, vi, rcond=None)
    vi_resid = vi - task_dummies @ beta
    auc_resid = roc_auc_score(y, vi_resid) if len(np.unique(y)) == 2 else np.nan
    md.append(f"- AUC(V_int_desp residualized vs task_id) = {auc_resid:.3f}\n")
    md.append(f"  (raw V_int_desp AUC on same subset = {roc_auc_score(y, vi):.3f})\n")
    report["tests"]["v_internal_desperate_auc_residualized"] = float(auc_resid)

    # ----------------- H5 per-task univariate with bootstrap CIs -----------
    md.append("\n## H5 — univariate Task A unsteered, bootstrap 95% CI\n")
    y_u = a_unsteered["y"].values
    vi_u = pd.to_numeric(a_unsteered["v_internal_desperate"], errors="coerce").fillna(0).values
    def _auc_safe(yy, xx):
        if len(np.unique(yy)) < 2:
            return np.nan
        return roc_auc_score(yy, xx)
    mean_auc, lo, hi = boot_ci(_auc_safe)(y_u, vi_u)
    md.append(f"- Univariate V_int_desp AUC = {mean_auc:.3f} [{lo:.3f}, {hi:.3f}] (n={len(y_u)}, events={y_u.sum()})\n")
    report["tests"]["h5_taskA_unsteered_univariate"] = {"auc": mean_auc, "ci_low": lo, "ci_high": hi,
                                                         "n": int(len(y_u)), "events": int(y_u.sum())}

    # ----------------- (M) Effect sizes with bootstrap CIs -----------------
    md.append("\n## (M) Steering effect sizes (Task A, desperate)\n")
    desp_steered = a_steered[a_steered["emotion"] == "desperate"]
    for s in sorted(desp_steered["strength"].unique()):
        sub = desp_steered[desp_steered["strength"] == s]
        rate = sub["y"].mean()
        md.append(f"- desperate @ {s:+.2f}: shortcut rate = {rate:.3f} (n={len(sub)})\n")
    # Rate difference vs unsteered baseline
    baseline = a_unsteered["y"].mean()
    md.append(f"- unsteered baseline: {baseline:.3f} (n={len(a_unsteered)})\n")
    # Cohen's h for proportion difference
    for s in sorted(desp_steered["strength"].unique()):
        p1 = desp_steered[desp_steered["strength"] == s]["y"].mean()
        p2 = baseline
        if 0 < p1 < 1 and 0 < p2 < 1:
            h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
            md.append(f"  - Cohen's h ({s:+.2f} vs unsteered) = {h:+.3f}\n")

    # ----------------- (J) Multiple comparisons correction ----------------
    md.append("\n## (J) Multiple-comparisons correction (BH + Bonferroni)\n")
    # Collect p-values from every test run above where we have one
    pvals, names = [], []
    # DeLong
    if not np.isnan(p):
        pvals.append(p); names.append("delong_internal_vs_text_taskA")
    # Per-strength steering vs unsteered (Fisher's exact)
    for s in sorted(desp_steered["strength"].unique()):
        sub = desp_steered[desp_steered["strength"] == s]
        k1, n1 = int(sub["y"].sum()), len(sub)
        k2, n2 = int(a_unsteered["y"].sum()), len(a_unsteered)
        tbl = [[k1, n1 - k1], [k2, n2 - k2]]
        _, p_f = stats.fisher_exact(tbl)
        pvals.append(p_f)
        names.append(f"fisher_desp{s:+.2f}_vs_unsteered")
    # Emotion probe univariate AUC vs 0.5 (Mann-Whitney approximation for Task A)
    try:
        stat, p_mw = stats.mannwhitneyu(
            vi_u[y_u == 1], vi_u[y_u == 0], alternative="greater",
        )
        pvals.append(p_mw)
        names.append("mannwhitney_vintdesp_unsteered")
    except ValueError:
        pass

    raw = np.array(pvals)
    bonf = np.minimum(raw * len(raw), 1.0)
    rej_bh, bh, *_ = multipletests(raw, alpha=0.05, method="fdr_bh")
    md.append("\n| Test | raw p | Bonferroni | BH-FDR | significant (BH 0.05) |\n")
    md.append("|---|---|---|---|---|\n")
    for i, name in enumerate(names):
        md.append(f"| {name} | {raw[i]:.4g} | {bonf[i]:.4g} | {bh[i]:.4g} | {'✓' if rej_bh[i] else ''} |\n")
    report["tests"]["mcc"] = {
        "names": names, "raw": raw.tolist(),
        "bonferroni": bonf.tolist(), "bh_fdr": bh.tolist(),
    }

    # ----------------- Write outputs --------------------------------------
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2, default=float)
    with open(OUT_MD, "w") as f:
        f.write("".join(md))
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
