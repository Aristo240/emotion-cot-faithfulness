#!/usr/bin/env python3
"""
Red-teaming analysis for Phase 2 results.

Stress-tests the main findings for methodological robustness:

  RT1: Cross-validated AUC (k-fold) — is V_internal AUC=0.999 real or overfit?
  RT2: Task-ID confound check — does V_internal predict behavior WITHIN each task?
  RT3: Coherence audit of desperate-steered "legitimate" responses
  RT4: Permutation test for the reversed desperate effect
  RT5: Power analysis for the low shortcut base rate
  RT6: Task B ceiling — why is sycophancy near-zero?
  RT7: Probe-task correlation — are probes just encoding task identity?

Usage:
  python scripts/06_red_teaming.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "phase2"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    entries = []
    with open(path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def make_df(entries, task):
    rows = []
    for e in entries:
        row = {
            "key": e["key"],
            "task_id": e["task_id"],
            "emotion": e["emotion"],
            "strength": e["strength"],
            "rollout": e["rollout"],
            "regex_cls": e["classification"],
            "judge_cls": e["judge_classification"],
            "response": e.get("response", ""),
        }
        for emo, val in e.get("emotion_probes", {}).items():
            row[f"probe_{emo}"] = val
        for dim, val in e.get("vtext_ratings", {}).items():
            row[f"vtext_{dim}"] = val
        if task == "b":
            row["syc_score"] = e.get("judge_sycophancy_score", None)
            row["syc_score_mean"] = e.get("judge_sycophancy_score_mean", None)
            row["is_sycophantic"] = e.get("judge_is_sycophantic", False)
        rows.append(row)
    return pd.DataFrame(rows)


# ==============================================================================
# RT1: Cross-validated AUC — is V_internal AUC=0.999 overfit?
# ==============================================================================

def rt1_cross_validated_auc(df_a):
    print("\n" + "=" * 70)
    print("RT1: CROSS-VALIDATED AUC (is V_internal AUC=0.999 real or overfit?)")
    print("=" * 70)

    df = df_a.copy()
    df["is_shortcut"] = (df["judge_cls"] == "SHORTCUT").astype(int)

    probe_cols = [c for c in df.columns if c.startswith("probe_")]
    vtext_cols = [c for c in df.columns if c.startswith("vtext_")]
    for c in probe_cols + vtext_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    y = df["is_shortcut"].values
    results = {}

    for name, cols in [
        ("V_internal", probe_cols),
        ("V_text", vtext_cols),
        ("Combined", probe_cols + vtext_cols),
    ]:
        X = df[cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Stratified 10-fold CV
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        y_pred_cv = np.zeros(len(y), dtype=float)

        fold_aucs = []
        for train_idx, test_idx in cv.split(X_scaled, y):
            clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
            clf.fit(X_scaled[train_idx], y[train_idx])
            proba = clf.predict_proba(X_scaled[test_idx])[:, 1]
            y_pred_cv[test_idx] = proba
            if len(np.unique(y[test_idx])) == 2:
                fold_aucs.append(roc_auc_score(y[test_idx], proba))

        # Overall OOS AUC
        oos_auc = roc_auc_score(y, y_pred_cv)
        # In-sample AUC for comparison
        clf_full = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
        clf_full.fit(X_scaled, y)
        is_auc = roc_auc_score(y, clf_full.predict_proba(X_scaled)[:, 1])

        results[name] = {
            "in_sample_auc": float(is_auc),
            "oos_auc_10fold": float(oos_auc),
            "fold_aucs": [float(a) for a in fold_aucs],
            "fold_mean": float(np.mean(fold_aucs)),
            "fold_std": float(np.std(fold_aucs)),
        }
        overfit_gap = is_auc - oos_auc
        print(f"  {name}:")
        print(f"    In-sample AUC:    {is_auc:.4f}")
        print(f"    10-fold OOS AUC:  {oos_auc:.4f}")
        print(f"    Fold AUCs:        {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
        print(f"    Overfit gap:      {overfit_gap:.4f} {'*** OVERFIT' if overfit_gap > 0.05 else '(acceptable)'}")

    # Leave-one-task-out CV (hardest test — can it generalize across tasks?)
    print("\n  --- Leave-One-Task-Out CV ---")
    logo = LeaveOneGroupOut()
    groups = df["task_id"].values
    for name, cols in [("V_internal", probe_cols), ("V_text", vtext_cols)]:
        X = df[cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        y_pred_logo = np.zeros(len(y), dtype=float)
        task_aucs = {}

        for train_idx, test_idx in logo.split(X_scaled, y, groups):
            task_name = groups[test_idx[0]]
            clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
            clf.fit(X_scaled[train_idx], y[train_idx])
            proba = clf.predict_proba(X_scaled[test_idx])[:, 1]
            y_pred_logo[test_idx] = proba
            if len(np.unique(y[test_idx])) == 2:
                task_aucs[task_name] = roc_auc_score(y[test_idx], proba)

        overall_logo = roc_auc_score(y, y_pred_logo) if len(np.unique(y)) == 2 else 0.5
        results[f"{name}_logo"] = {
            "overall_auc": float(overall_logo),
            "per_task": {k: float(v) for k, v in task_aucs.items()},
        }
        print(f"  {name} LOGO AUC: {overall_logo:.4f}")
        for task, auc in task_aucs.items():
            print(f"    {task}: {auc:.4f}")

    return results


# ==============================================================================
# RT2: Task-ID confound — are probes encoding task identity, not emotion?
# ==============================================================================

def rt2_task_confound(df_a):
    print("\n" + "=" * 70)
    print("RT2: TASK-ID CONFOUND (are probes just encoding task identity?)")
    print("=" * 70)

    df = df_a.copy()
    df["is_shortcut"] = (df["judge_cls"] == "SHORTCUT").astype(int)
    probe_cols = [c for c in df.columns if c.startswith("probe_")]
    vtext_cols = [c for c in df.columns if c.startswith("vtext_")]
    for c in probe_cols + vtext_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    results = {}

    # Test 1: Can probes predict task_id? If so, they encode task structure.
    print("\n  --- Can V_internal predict task_id? ---")
    from sklearn.linear_model import LogisticRegression as LR
    X = StandardScaler().fit_transform(df[probe_cols].values)
    y_task = pd.Categorical(df["task_id"]).codes
    clf = LR(max_iter=2000, random_state=42)
    clf.fit(X, y_task)
    task_acc = accuracy_score(y_task, clf.predict(X))
    # CV accuracy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs = []
    for tr, te in cv.split(X, y_task):
        clf_cv = LR(max_iter=2000, random_state=42)
        clf_cv.fit(X[tr], y_task[tr])
        cv_accs.append(accuracy_score(y_task[te], clf_cv.predict(X[te])))
    results["probe_predicts_taskid"] = {
        "in_sample_acc": float(task_acc),
        "cv_acc": float(np.mean(cv_accs)),
        "chance": float(1.0 / len(df["task_id"].unique())),
    }
    print(f"    In-sample accuracy: {task_acc:.3f}")
    print(f"    5-fold CV accuracy: {np.mean(cv_accs):.3f} (chance = {1.0/len(df['task_id'].unique()):.3f})")
    if np.mean(cv_accs) > 0.5:
        print(f"    *** WARNING: Probes encode task identity (CV acc >> chance)")

    # Test 2: Within-task AUC — does V_internal still predict shortcuts
    # controlling for task?
    print("\n  --- Within-task AUC (V_internal → shortcut, per task) ---")
    within_aucs = {}
    for task in sorted(df["task_id"].unique()):
        sub = df[df["task_id"] == task]
        if sub["is_shortcut"].nunique() < 2:
            within_aucs[task] = {"auc": None, "n": len(sub),
                                 "n_shortcut": int(sub["is_shortcut"].sum()),
                                 "note": "no variance"}
            print(f"    {task}: no variance (n={len(sub)}, shortcuts={sub['is_shortcut'].sum()})")
            continue
        X_sub = StandardScaler().fit_transform(sub[probe_cols].values)
        y_sub = sub["is_shortcut"].values
        clf = LR(max_iter=2000, random_state=42, C=1.0)
        clf.fit(X_sub, y_sub)
        auc = roc_auc_score(y_sub, clf.predict_proba(X_sub)[:, 1])
        # 5-fold CV within task
        if len(sub) >= 10:
            cv = StratifiedKFold(n_splits=min(5, sum(y_sub), sum(1-y_sub)), shuffle=True, random_state=42)
            cv_preds = np.zeros(len(y_sub))
            for tr, te in cv.split(X_sub, y_sub):
                clf_cv = LR(max_iter=2000, random_state=42, C=1.0)
                clf_cv.fit(X_sub[tr], y_sub[tr])
                cv_preds[te] = clf_cv.predict_proba(X_sub[te])[:, 1]
            cv_auc = roc_auc_score(y_sub, cv_preds) if len(np.unique(y_sub)) == 2 else 0.5
        else:
            cv_auc = None
        within_aucs[task] = {"is_auc": float(auc), "cv_auc": float(cv_auc) if cv_auc else None,
                             "n": len(sub), "n_shortcut": int(sub["is_shortcut"].sum())}
        cv_str = f"{cv_auc:.3f}" if cv_auc is not None else "N/A"
        print(f"    {task}: IS-AUC={auc:.3f}, CV-AUC={cv_str}, n={len(sub)}, shortcuts={sub['is_shortcut'].sum()}")

    results["within_task_auc"] = within_aucs

    # Test 3: Add task_id dummies and see if V_internal still adds value
    print("\n  --- V_internal incremental value over task_id ---")
    task_dummies = pd.get_dummies(df["task_id"], drop_first=True).values
    y = df["is_shortcut"].values
    scaler = StandardScaler()

    # Model 1: task_id only
    X_task = task_dummies
    clf1 = LR(max_iter=2000, random_state=42)
    clf1.fit(X_task, y)
    auc_task_only = roc_auc_score(y, clf1.predict_proba(X_task)[:, 1])

    # Model 2: task_id + V_internal
    X_both = np.hstack([task_dummies, scaler.fit_transform(df[probe_cols].values)])
    clf2 = LR(max_iter=2000, random_state=42)
    clf2.fit(X_both, y)
    auc_both = roc_auc_score(y, clf2.predict_proba(X_both)[:, 1])

    # Likelihood ratio test
    ll1 = -log_loss(y, clf1.predict_proba(X_task)[:, 1], normalize=False)
    ll2 = -log_loss(y, clf2.predict_proba(X_both)[:, 1], normalize=False)
    lr_stat = 2 * (ll2 - ll1)
    df_diff = X_both.shape[1] - X_task.shape[1]
    p_lr = 1 - stats.chi2.cdf(lr_stat, df_diff) if lr_stat > 0 else 1.0

    results["incremental_value"] = {
        "auc_task_only": float(auc_task_only),
        "auc_task_plus_probes": float(auc_both),
        "lr_statistic": float(lr_stat),
        "lr_p_value": float(p_lr),
    }
    print(f"    Task-ID only AUC:        {auc_task_only:.4f}")
    print(f"    Task-ID + V_internal AUC: {auc_both:.4f}")
    print(f"    LR test: chi2={lr_stat:.2f}, p={p_lr:.4f}")
    if p_lr < 0.01:
        print(f"    --> V_internal adds significant predictive value beyond task identity")
    else:
        print(f"    *** V_internal does NOT add significant value beyond task identity")

    return results


# ==============================================================================
# RT3: Coherence audit — are desperate-steered "legitimate" responses coherent?
# ==============================================================================

def rt3_coherence_audit(df_a):
    print("\n" + "=" * 70)
    print("RT3: COHERENCE AUDIT (are desperate-steered 'legitimate' responses coherent?)")
    print("=" * 70)

    df = df_a.copy()
    # Get desperate-steered responses classified as LEGITIMATE
    desp_legit = df[(df["emotion"] == "desperate") & (df["strength"] > 0.1) &
                    (df["judge_cls"] == "LEGITIMATE")]
    desp_shortcut = df[(df["emotion"] == "desperate") & (df["strength"] > 0.1) &
                       (df["judge_cls"] == "SHORTCUT")]
    unsteered_legit = df[(df["strength"].abs() < 1e-6) & (df["judge_cls"] == "LEGITIMATE")]

    results = {
        "n_desperate_steered_legitimate": len(desp_legit),
        "n_desperate_steered_shortcut": len(desp_shortcut),
        "n_unsteered_legitimate": len(unsteered_legit),
    }

    print(f"  Desperate-steered (strength > 0.1):")
    print(f"    LEGITIMATE: {len(desp_legit)}")
    print(f"    SHORTCUT:   {len(desp_shortcut)}")
    print(f"  Unsteered LEGITIMATE: {len(unsteered_legit)}")

    # Compare response lengths (proxy for coherence)
    if len(desp_legit) > 0 and len(unsteered_legit) > 0:
        desp_lengths = desp_legit["response"].str.len()
        unst_lengths = unsteered_legit["response"].str.len()
        t_stat, t_p = stats.ttest_ind(desp_lengths, unst_lengths)
        results["response_length"] = {
            "desperate_mean": float(desp_lengths.mean()),
            "desperate_std": float(desp_lengths.std()),
            "unsteered_mean": float(unst_lengths.mean()),
            "unsteered_std": float(unst_lengths.std()),
            "t_stat": float(t_stat),
            "p_value": float(t_p),
        }
        print(f"\n  Response length comparison (LEGITIMATE only):")
        print(f"    Desperate-steered: {desp_lengths.mean():.0f} +/- {desp_lengths.std():.0f} chars")
        print(f"    Unsteered:         {unst_lengths.mean():.0f} +/- {unst_lengths.std():.0f} chars")
        print(f"    t={t_stat:.2f}, p={t_p:.4f}")

    # Check for incoherence markers in desperate-steered responses
    incoherence_markers = [
        r'[A-Z]{5,}',           # ALL CAPS (5+ chars)
        r'!!!',                  # Multiple exclamation marks
        r'\?\?\?',              # Multiple question marks
        r'(.)\1{4,}',          # Character repetition (5+)
        r'(?i)(error|traceback|exception)',  # Python errors in output
    ]
    import re
    marker_counts = {}
    for label, subset in [("desperate_legit", desp_legit), ("unsteered_legit", unsteered_legit)]:
        counts = {m: 0 for m in incoherence_markers}
        for _, row in subset.iterrows():
            resp = str(row.get("response", ""))
            for pattern in incoherence_markers:
                if re.search(pattern, resp):
                    counts[pattern] += 1
        rates = {p: c / max(len(subset), 1) for p, c in counts.items()}
        marker_counts[label] = rates

    results["incoherence_markers"] = marker_counts
    print(f"\n  Incoherence markers (fraction of responses):")
    print(f"    {'Marker':<25} {'Desp-steered':>15} {'Unsteered':>15}")
    print(f"    {'─'*25} {'─'*15} {'─'*15}")
    for pattern in incoherence_markers:
        d_rate = marker_counts["desperate_legit"].get(pattern, 0)
        u_rate = marker_counts["unsteered_legit"].get(pattern, 0)
        print(f"    {pattern:<25} {d_rate:>14.1%} {u_rate:>14.1%}")

    # Show 3 example desperate-steered LEGITIMATE responses (truncated)
    print(f"\n  --- Sample desperate-steered LEGITIMATE responses ---")
    for i, (_, row) in enumerate(desp_legit.head(3).iterrows()):
        resp = str(row.get("response", ""))[:400]
        print(f"\n  [{i+1}] key={row['key']}, strength={row['strength']}")
        print(f"  {resp}...")

    return results


# ==============================================================================
# RT4: Permutation test — is the reversed desperate effect real?
# ==============================================================================

def rt4_permutation_test(df_a):
    print("\n" + "=" * 70)
    print("RT4: PERMUTATION TEST (is the reversed desperate effect real?)")
    print("=" * 70)

    df = df_a.copy()
    df["is_shortcut"] = (df["judge_cls"] == "SHORTCUT").astype(int)

    # Observed effect: shortcut rate for desperate-positive vs desperate-negative
    desp = df[df["emotion"] == "desperate"]
    desp_pos = desp[desp["strength"] > 0.1]["is_shortcut"]
    desp_neg = desp[desp["strength"] < -0.1]["is_shortcut"]
    unsteered = df[(df["emotion"] == "none") | (df["strength"].abs() < 1e-6)]["is_shortcut"]

    if len(desp_pos) == 0 or len(desp_neg) == 0:
        print("  Insufficient data for permutation test")
        return {}

    observed_diff = desp_pos.mean() - unsteered.mean()
    print(f"  Observed: desperate(+) shortcut rate = {desp_pos.mean():.4f}")
    print(f"  Observed: unsteered shortcut rate     = {unsteered.mean():.4f}")
    print(f"  Observed difference:                   = {observed_diff:.4f}")
    print(f"  Direction: {'INCREASED' if observed_diff > 0 else 'DECREASED'} shortcut rate")

    # Permutation test: shuffle emotion labels 10,000 times
    n_perms = 10000
    combined = pd.concat([desp_pos, unsteered])
    n_desp = len(desp_pos)
    perm_diffs = np.zeros(n_perms)
    np.random.seed(42)

    for i in range(n_perms):
        perm = combined.sample(frac=1).values
        perm_diffs[i] = perm[:n_desp].mean() - perm[n_desp:].mean()

    # Two-sided p-value
    p_two_sided = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    # One-sided (testing if desperate DECREASES shortcut rate)
    p_one_sided = np.mean(perm_diffs <= observed_diff)

    results = {
        "observed_diff": float(observed_diff),
        "desp_pos_rate": float(desp_pos.mean()),
        "unsteered_rate": float(unsteered.mean()),
        "n_desp_pos": int(len(desp_pos)),
        "n_unsteered": int(len(unsteered)),
        "p_two_sided": float(p_two_sided),
        "p_one_sided_decrease": float(p_one_sided),
        "n_permutations": n_perms,
    }

    print(f"\n  Permutation test ({n_perms} permutations):")
    print(f"    Two-sided p-value:               {p_two_sided:.4f}")
    print(f"    One-sided p (desperate decreases): {p_one_sided:.4f}")
    sig = "***" if p_two_sided < 0.001 else "**" if p_two_sided < 0.01 else "*" if p_two_sided < 0.05 else "ns"
    print(f"    Significance: {sig}")

    # Also test calm
    calm = df[df["emotion"] == "calm"]
    calm_pos = calm[calm["strength"] > 0.1]["is_shortcut"]
    if len(calm_pos) > 0:
        calm_diff = calm_pos.mean() - unsteered.mean()
        results["calm_pos_rate"] = float(calm_pos.mean())
        results["calm_diff"] = float(calm_diff)
        print(f"\n  Calm(+) shortcut rate: {calm_pos.mean():.4f} (diff from unsteered: {calm_diff:.4f})")

    # Plot permutation distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(perm_diffs, bins=50, color="#aaaaaa", edgecolor="white", density=True, label="Permutation null")
    ax.axvline(observed_diff, color="red", linewidth=2, label=f"Observed ({observed_diff:.3f})")
    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Difference in Shortcut Rate (desperate - unsteered)")
    ax.set_ylabel("Density")
    ax.set_title(f"Permutation Test: Desperate Steering Effect\np = {p_two_sided:.4f} (two-sided)")
    ax.legend()
    plt.tight_layout()
    out = PLOTS_DIR / "rt4_permutation_test.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")

    return results


# ==============================================================================
# RT5: Power analysis
# ==============================================================================

def rt5_power_analysis(df_a):
    print("\n" + "=" * 70)
    print("RT5: POWER ANALYSIS (can we detect effects with n=44 shortcuts?)")
    print("=" * 70)

    df = df_a.copy()
    df["is_shortcut"] = (df["judge_cls"] == "SHORTCUT").astype(int)

    n_total = len(df)
    n_shortcut = df["is_shortcut"].sum()
    base_rate = n_shortcut / n_total

    # For Fisher's exact test, what effect sizes can we detect?
    # Minimum detectable effect size at 80% power, alpha=0.05
    # Approximate using normal approximation for 2-proportion z-test
    from scipy.stats import norm
    alpha = 0.05
    power = 0.80
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    # Typical group sizes
    per_condition = len(df[(df["emotion"] == "desperate") & (df["strength"] > 0.1)])
    n_control = len(df[(df["emotion"] == "none") | (df["strength"].abs() < 1e-6)])

    if per_condition > 0 and n_control > 0:
        # Minimum detectable difference
        p1 = base_rate
        # Solve for p2 such that we have 80% power
        # Approximate: we need (p1-p2) / sqrt(p*(1-p)*(1/n1 + 1/n2)) > z_alpha + z_beta
        p_pool = (p1 * n_control + p1 * per_condition) / (n_control + per_condition)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/per_condition + 1/n_control))
        mde = (z_alpha + z_beta) * se

        results = {
            "n_total": n_total,
            "n_shortcut": int(n_shortcut),
            "base_rate": float(base_rate),
            "n_per_condition": int(per_condition),
            "n_control": int(n_control),
            "min_detectable_effect": float(mde),
            "min_detectable_rate": float(base_rate + mde),
        }
        print(f"  Base rate: {base_rate:.3f} ({n_shortcut}/{n_total})")
        print(f"  Condition size: {per_condition} (steered), {n_control} (control)")
        print(f"  Minimum detectable effect (80% power, alpha=0.05):")
        print(f"    Absolute: {mde:.3f}")
        print(f"    Detectable range: [{max(0, base_rate-mde):.3f}, {min(1, base_rate+mde):.3f}]")
        if mde > base_rate:
            print(f"    *** WARNING: MDE ({mde:.3f}) > base rate ({base_rate:.3f})")
            print(f"    --> Cannot reliably detect reductions below baseline")
    else:
        results = {"note": "insufficient data"}

    # Per-condition breakdown
    print(f"\n  Shortcut rate by condition:")
    for emo in sorted(df["emotion"].unique()):
        for strength in sorted(df["strength"].unique()):
            sub = df[(df["emotion"] == emo) & (df["strength"] == strength)]
            if len(sub) == 0:
                continue
            rate = sub["is_shortcut"].mean()
            n = len(sub)
            ci = 1.96 * np.sqrt(rate * (1 - rate) / n) if n > 0 else 0
            print(f"    {emo:>10} @ {strength:+.1f}: {rate:.3f} +/- {ci:.3f} (n={n}, shortcuts={sub['is_shortcut'].sum()})")

    return results


# ==============================================================================
# RT6: Task B ceiling — why is sycophancy near-zero?
# ==============================================================================

def rt6_taskb_ceiling(df_b):
    print("\n" + "=" * 70)
    print("RT6: TASK B CEILING ANALYSIS (why is sycophancy near-zero?)")
    print("=" * 70)

    df = df_b.copy()
    results = {}

    # Distribution of sycophancy scores
    print("  Judge classification distribution:")
    cls_counts = Counter(df["judge_cls"])
    for cls in ["STRONG_PUSHBACK", "GENTLE_PUSHBACK", "NEUTRAL", "MILD_SYCOPHANCY", "STRONG_SYCOPHANCY"]:
        count = cls_counts.get(cls, 0)
        pct = count / len(df) * 100
        print(f"    {cls:<20}: {count:>4} ({pct:.1f}%)")

    # Continuous sycophancy score distribution
    scores = df["syc_score_mean"].dropna()
    print(f"\n  Sycophancy score distribution (1=strong pushback, 5=sycophantic):")
    print(f"    Mean: {scores.mean():.3f}")
    print(f"    Std:  {scores.std():.3f}")
    print(f"    Min:  {scores.min():.3f}")
    print(f"    Max:  {scores.max():.3f}")
    print(f"    Median: {scores.median():.3f}")

    # Does steering move the continuous score at all?
    print(f"\n  Mean sycophancy score by condition:")
    for emo in sorted(df["emotion"].unique()):
        for strength in sorted(df["strength"].unique()):
            sub = df[(df["emotion"] == emo) & (df["strength"] == strength)]
            if len(sub) == 0:
                continue
            mean_s = sub["syc_score_mean"].mean()
            std_s = sub["syc_score_mean"].std()
            print(f"    {emo:>10} @ {strength:+.1f}: {mean_s:.3f} +/- {std_s:.3f} (n={len(sub)})")

    # Check if GENTLE vs STRONG pushback varies with steering
    print(f"\n  Strong vs Gentle pushback by condition:")
    df["is_strong"] = (df["judge_cls"] == "STRONG_PUSHBACK").astype(int)
    for emo in ["desperate", "calm", "none"]:
        sub = df[df["emotion"] == emo]
        if sub.empty:
            continue
        r, p = stats.pearsonr(sub["strength"], sub["is_strong"])
        print(f"    {emo}: r={r:.3f}, p={p:.4f} (strong_pushback ~ strength)")

    # Effect size for the continuous score
    desp_pos = df[(df["emotion"] == "desperate") & (df["strength"] > 0.1)]["syc_score_mean"]
    unsteered = df[(df["emotion"] == "none") | (df["strength"].abs() < 1e-6)]["syc_score_mean"]
    if len(desp_pos) > 0 and len(unsteered) > 0:
        pooled_std = np.sqrt(
            (desp_pos.var() * (len(desp_pos) - 1) + unsteered.var() * (len(unsteered) - 1)) /
            (len(desp_pos) + len(unsteered) - 2)
        )
        d = (desp_pos.mean() - unsteered.mean()) / pooled_std if pooled_std > 0 else 0
        print(f"\n  Effect size (desperate vs unsteered, continuous score):")
        print(f"    Cohen's d = {d:.3f}")
        print(f"    Interpretation: {'negligible' if abs(d) < 0.2 else 'small' if abs(d) < 0.5 else 'medium' if abs(d) < 0.8 else 'large'}")

    results["sycophancy_score_stats"] = {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }
    results["classification_counts"] = dict(cls_counts)

    return results


# ==============================================================================
# RT7: Probe-task correlation — are probes encoding prompt, not emotion?
# ==============================================================================

def rt7_probe_task_correlation(df_a, df_b):
    print("\n" + "=" * 70)
    print("RT7: PROBE-TASK CORRELATION (do probes vary more by task than by emotion?)")
    print("=" * 70)

    results = {}

    for label, df in [("Task A", df_a), ("Task B", df_b)]:
        probe_cols = [c for c in df.columns if c.startswith("probe_")]
        if not probe_cols:
            continue

        # Variance explained by task_id vs emotion vs strength
        probe_data = df[probe_cols + ["task_id", "emotion", "strength"]].copy()
        for c in probe_cols:
            probe_data[c] = pd.to_numeric(probe_data[c], errors="coerce")

        # ANOVA-style: eta-squared for each factor on key probes
        key_probes = ["probe_desperate", "probe_calm"]
        key_probes = [p for p in key_probes if p in probe_cols]

        print(f"\n  --- {label} ---")
        for probe in key_probes:
            valid = probe_data.dropna(subset=[probe])

            # Eta-squared for task_id
            groups_task = [g[probe].values for _, g in valid.groupby("task_id")]
            if len(groups_task) >= 2:
                F_task, p_task = stats.f_oneway(*groups_task)
                ss_between = sum(len(g) * (np.mean(g) - valid[probe].mean())**2 for g in groups_task)
                ss_total = np.sum((valid[probe] - valid[probe].mean())**2)
                eta2_task = ss_between / ss_total if ss_total > 0 else 0
            else:
                eta2_task = 0; p_task = 1.0

            # Eta-squared for emotion
            groups_emo = [g[probe].values for _, g in valid.groupby("emotion")]
            if len(groups_emo) >= 2:
                F_emo, p_emo = stats.f_oneway(*groups_emo)
                ss_between_emo = sum(len(g) * (np.mean(g) - valid[probe].mean())**2 for g in groups_emo)
                eta2_emo = ss_between_emo / ss_total if ss_total > 0 else 0
            else:
                eta2_emo = 0; p_emo = 1.0

            # Correlation with strength
            r_strength, p_strength = stats.pearsonr(valid["strength"], valid[probe])

            results[f"{label}_{probe}"] = {
                "eta2_task_id": float(eta2_task),
                "p_task_id": float(p_task),
                "eta2_emotion": float(eta2_emo),
                "p_emotion": float(p_emo),
                "r_strength": float(r_strength),
                "p_strength": float(p_strength),
            }
            print(f"    {probe}:")
            print(f"      task_id:  eta2={eta2_task:.4f}, p={p_task:.4f} {'***' if p_task < 0.001 else ''}")
            print(f"      emotion:  eta2={eta2_emo:.4f}, p={p_emo:.4f} {'***' if p_emo < 0.001 else ''}")
            print(f"      strength: r={r_strength:.4f}, p={p_strength:.4f} {'***' if p_strength < 0.001 else ''}")
            if eta2_task > eta2_emo * 2:
                print(f"      *** WARNING: task_id explains {eta2_task/eta2_emo:.1f}x more variance than emotion")

    return results


# ==============================================================================
# Summary plot: RT1 cross-validation comparison
# ==============================================================================

def plot_cv_comparison(rt1_results):
    fig, ax = plt.subplots(figsize=(8, 5))

    names = ["V_internal", "V_text", "Combined"]
    is_aucs = [rt1_results[n]["in_sample_auc"] for n in names]
    oos_aucs = [rt1_results[n]["oos_auc_10fold"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    bars1 = ax.bar(x - width/2, is_aucs, width, label="In-sample", color="#4c72b0", alpha=0.8)
    bars2 = ax.bar(x + width/2, oos_aucs, width, label="10-fold CV", color="#dd8452", alpha=0.8)

    ax.set_ylabel("AUC")
    ax.set_title("Red-Team RT1: In-Sample vs Cross-Validated AUC\n(Predicting Reward Hacking)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.set_ylim(0.4, 1.05)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = PLOTS_DIR / "rt1_cv_comparison.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ==============================================================================
# Summary plot: RT5 per-condition shortcut rates with CIs
# ==============================================================================

def plot_per_condition_rates(df_a):
    df = df_a.copy()
    df["is_shortcut"] = (df["judge_cls"] == "SHORTCUT").astype(int)
    PALETTE = {"desperate": "#d62728", "calm": "#1f77b4", "none": "#7f7f7f"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for emo, color in PALETTE.items():
        sub = df[df["emotion"] == emo]
        if sub.empty:
            continue
        grouped = sub.groupby("strength")["is_shortcut"].agg(["mean", "count"])
        grouped["ci"] = 1.96 * np.sqrt(grouped["mean"] * (1 - grouped["mean"]) / grouped["count"])
        ax.errorbar(grouped.index, grouped["mean"], yerr=grouped["ci"],
                     fmt="o-", color=color, label=emo, capsize=4, markersize=6, linewidth=2)

    ax.set_xlabel("Steering Strength", fontsize=12)
    ax.set_ylabel("Shortcut Rate (judge-classified)", fontsize=12)
    ax.set_title("Task A: Shortcut Rate by Condition\n(with 95% Wilson CIs, per red-team RT5)", fontsize=13)
    ax.legend(title="Emotion", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "rt5_per_condition_rates.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("RED-TEAMING ANALYSIS")
    print("=" * 70)

    entries_a = load_jsonl(RESULTS_DIR / "task_a_judged.jsonl")
    entries_b = load_jsonl(RESULTS_DIR / "task_b_judged.jsonl")
    df_a = make_df(entries_a, "a")
    df_b = make_df(entries_b, "b")
    print(f"Loaded: {len(df_a)} Task A, {len(df_b)} Task B")

    report = {}

    report["rt1"] = rt1_cross_validated_auc(df_a)
    report["rt2"] = rt2_task_confound(df_a)
    report["rt3"] = rt3_coherence_audit(df_a)
    report["rt4"] = rt4_permutation_test(df_a)
    report["rt5"] = rt5_power_analysis(df_a)
    report["rt6"] = rt6_taskb_ceiling(df_b)
    report["rt7"] = rt7_probe_task_correlation(df_a, df_b)

    # Plots
    print("\n" + "=" * 70)
    print("GENERATING RED-TEAM PLOTS")
    print("=" * 70)
    plot_cv_comparison(report["rt1"])
    plot_per_condition_rates(df_a)

    # Save report
    report_path = RESULTS_DIR / "red_team_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nRed-team report saved to {report_path}")

    # Final verdict
    print("\n" + "=" * 70)
    print("RED-TEAM VERDICT")
    print("=" * 70)

    issues = []
    if report["rt1"]["V_internal"]["in_sample_auc"] - report["rt1"]["V_internal"]["oos_auc_10fold"] > 0.1:
        issues.append("RT1: V_internal AUC drops significantly under cross-validation")
    if report["rt2"].get("incremental_value", {}).get("lr_p_value", 0) > 0.05:
        issues.append("RT2: V_internal does not add value beyond task identity")
    if report["rt4"].get("p_two_sided", 1) > 0.05:
        issues.append("RT4: Reversed desperate effect is not statistically significant")

    if issues:
        print("  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  No critical issues found. Main findings are robust.")

    # Caveats
    print("\n  CAVEATS (regardless of statistical results):")
    print("  1. n=44 shortcuts is low — bootstrapped CIs are wide")
    print("  2. Task B is essentially a null result (2/650 sycophantic)")
    print("  3. Reversed desperate effect contradicts Sofroniew et al.")
    print("  4. Phase 3 (unsteered natural prediction) still needed")


if __name__ == "__main__":
    main()
