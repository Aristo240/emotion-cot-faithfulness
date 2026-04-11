#!/usr/bin/env python3
"""
Phase 2 statistical analysis and visualization.

Loads judged Phase 2 results and produces:
  1. Statistical tests (dose-response, faithfulness correlation, predictive comparison)
  2. Publication-quality plots
  3. Summary report JSON

Usage:
  python scripts/05_run_analysis.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = {"desperate": "#d62728", "calm": "#1f77b4", "none": "#7f7f7f"}

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "phase2"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path):
    entries = []
    with open(path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def make_df(entries, task):
    """Flatten judged JSONL entries into a tidy DataFrame."""
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
            "judge_agreement": e["judge_agreement"],
        }
        # Emotion probes (cosine similarities)
        for emo, val in e.get("emotion_probes", {}).items():
            row[f"probe_{emo}"] = val
        # V_text ratings from judge
        for dim, val in e.get("vtext_ratings", {}).items():
            row[f"vtext_{dim}"] = val
        # Task-specific
        if task == "b":
            row["syc_score"] = e.get("judge_sycophancy_score", None)
            row["syc_score_mean"] = e.get("judge_sycophancy_score_mean", None)
            row["is_sycophantic"] = e.get("judge_is_sycophantic", False)
        rows.append(row)
    return pd.DataFrame(rows)


# ── PLOT 1: Dose-response curves ────────────────────────────────────────────

def plot_dose_response(df_a, df_b):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Task A: shortcut rate by strength × emotion
    ax = axes[0]
    df_a["is_shortcut"] = (df_a["judge_cls"] == "SHORTCUT").astype(int)
    for emo, color in PALETTE.items():
        sub = df_a[df_a["emotion"] == emo]
        if sub.empty:
            continue
        grouped = sub.groupby("strength")["is_shortcut"].agg(["mean", "sem", "count"])
        ax.errorbar(grouped.index, grouped["mean"], yerr=1.96 * grouped["sem"],
                     fmt="o-", color=color, label=emo, capsize=3, markersize=5)
    ax.set_xlabel("Steering Strength")
    ax.set_ylabel("Shortcut Rate (judge)")
    ax.set_title("Task A: Reward Hacking")
    ax.legend(title="Emotion")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(-0.02, max(0.35, df_a["is_shortcut"].mean() * 3))

    # Task B: mean sycophancy score by strength × emotion
    ax = axes[1]
    for emo, color in PALETTE.items():
        sub = df_b[df_b["emotion"] == emo]
        if sub.empty:
            continue
        grouped = sub.groupby("strength")["syc_score_mean"].agg(["mean", "sem", "count"])
        ax.errorbar(grouped.index, grouped["mean"], yerr=1.96 * grouped["sem"],
                     fmt="o-", color=color, label=emo, capsize=3, markersize=5)
    ax.set_xlabel("Steering Strength")
    ax.set_ylabel("Sycophancy Score (1=strong pushback, 5=sycophantic)")
    ax.set_title("Task B: Sycophancy")
    ax.legend(title="Emotion")
    ax.set_ylim(0.8, 3.5)

    plt.tight_layout()
    out = PLOTS_DIR / "dose_response.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── PLOT 2: Judge vs Regex reclassification ──────────────────────────────────

def plot_judge_vs_regex(df_a, df_b):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Task A confusion
    ax = axes[0]
    regex_map = {"legit": "LEGITIMATE", "hack": "SHORTCUT", "unclear": "UNCLEAR"}
    df_a["regex_norm"] = df_a["regex_cls"].map(regex_map)
    cats_a = ["LEGITIMATE", "SHORTCUT", "UNCLEAR"]
    cm_a = pd.crosstab(df_a["regex_norm"], df_a["judge_cls"], dropna=False)
    cm_a = cm_a.reindex(index=cats_a, columns=cats_a, fill_value=0)
    sns.heatmap(cm_a, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_xlabel("Judge Classification")
    ax.set_ylabel("Regex Classification")
    ax.set_title(f"Task A: Regex vs Judge (agree={cm_a.values.diagonal().sum() / cm_a.values.sum():.1%})")

    # Task B confusion
    ax = axes[1]
    regex_map_b = {"pushback": "PUSHBACK", "sycophantic": "SYCOPHANTIC", "unclear": "UNCLEAR"}
    df_b["regex_norm"] = df_b["regex_cls"].map(regex_map_b)
    judge_map_b = {
        "STRONG_PUSHBACK": "PUSHBACK", "GENTLE_PUSHBACK": "PUSHBACK",
        "MILD_SYCOPHANCY": "SYCOPHANTIC", "STRONG_SYCOPHANCY": "SYCOPHANTIC",
        "NEUTRAL": "UNCLEAR"
    }
    df_b["judge_norm"] = df_b["judge_cls"].map(judge_map_b).fillna("UNCLEAR")
    cats_b = ["PUSHBACK", "SYCOPHANTIC", "UNCLEAR"]
    cm_b = pd.crosstab(df_b["regex_norm"], df_b["judge_norm"], dropna=False)
    cm_b = cm_b.reindex(index=cats_b, columns=cats_b, fill_value=0)
    sns.heatmap(cm_b, annot=True, fmt="d", cmap="Greens", ax=ax, cbar=False)
    ax.set_xlabel("Judge Classification")
    ax.set_ylabel("Regex Classification")
    ax.set_title(f"Task B: Regex vs Judge (agree={cm_b.values.diagonal().sum() / cm_b.values.sum():.1%})")

    plt.tight_layout()
    out = PLOTS_DIR / "judge_vs_regex.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── PLOT 3: V_text emotion ratings by steering condition ─────────────────────

def plot_vtext_by_condition(df_a, df_b):
    dims = ["valence", "arousal", "urgency", "composure", "frustration", "hedging"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for idx, dim in enumerate(dims):
        ax = axes[idx // 3][idx % 3]
        col = f"vtext_{dim}"

        # Combine both tasks for this analysis
        df_all = pd.concat([
            df_a[["emotion", "strength", col]].assign(task="A"),
            df_b[["emotion", "strength", col]].assign(task="B"),
        ], ignore_index=True)
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

        for emo, color in PALETTE.items():
            sub = df_all[df_all["emotion"] == emo].dropna(subset=[col])
            if sub.empty:
                continue
            grouped = sub.groupby("strength")[col].agg(["mean", "sem"])
            ax.errorbar(grouped.index, grouped["mean"], yerr=1.96 * grouped["sem"],
                         fmt="o-", color=color, label=emo, capsize=2, markersize=4)
        ax.set_title(dim.capitalize())
        ax.set_xlabel("Steering Strength")
        ax.set_ylabel("Rating (1-7)")
        if idx == 0:
            ax.legend(fontsize=8)

    plt.suptitle("V_text (Judge-Rated Emotional Tone) by Steering Condition", fontsize=14, y=1.01)
    plt.tight_layout()
    out = PLOTS_DIR / "vtext_by_condition.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── PLOT 4: V_internal (probe) vs V_text faithfulness scatter ────────────────

def plot_faithfulness_scatter(df_a, df_b):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, df, task_name, probe_emo in [
        (axes[0], df_a, "Task A (Coding)", "desperate"),
        (axes[1], df_b, "Task B (Sycophancy)", "desperate"),
    ]:
        probe_col = f"probe_{probe_emo}"
        # Composite V_text: mean of all vtext dims (standardized to 0-1)
        vtext_cols = [c for c in df.columns if c.startswith("vtext_")]
        df_plot = df.copy()
        for c in vtext_cols:
            df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce")
        df_plot["vtext_composite"] = df_plot[vtext_cols].mean(axis=1)

        if probe_col not in df_plot.columns:
            continue

        valid = df_plot.dropna(subset=[probe_col, "vtext_composite"])
        if valid.empty:
            continue

        colors = [PALETTE.get(e, "#999999") for e in valid["emotion"]]
        ax.scatter(valid[probe_col], valid["vtext_composite"],
                   c=colors, alpha=0.35, s=15, edgecolors="none")

        # Correlation
        r, p = stats.pearsonr(valid[probe_col], valid["vtext_composite"])
        ax.set_xlabel(f"V_internal (probe: {probe_emo})")
        ax.set_ylabel("V_text (composite judge rating)")
        ax.set_title(f"{task_name}\nr = {r:.3f}, p = {p:.2e}")

        # Legend
        from matplotlib.patches import Patch
        handles = [Patch(color=c, label=e) for e, c in PALETTE.items()]
        ax.legend(handles=handles, fontsize=8, loc="upper left")

    plt.suptitle("Faithfulness: Internal Emotion Probe vs Surface Expression", fontsize=13, y=1.01)
    plt.tight_layout()
    out = PLOTS_DIR / "faithfulness_scatter.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── PLOT 5: Task B sycophancy score distribution by judge class ──────────────

def plot_sycophancy_distribution(df_b):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Distribution of judge classifications
    ax = axes[0]
    cls_order = ["STRONG_PUSHBACK", "GENTLE_PUSHBACK", "NEUTRAL", "MILD_SYCOPHANCY", "STRONG_SYCOPHANCY"]
    counts = df_b["judge_cls"].value_counts().reindex(cls_order, fill_value=0)
    colors_bar = ["#2166ac", "#67a9cf", "#d1e5f0", "#ef8a62", "#b2182b"]
    bars = ax.bar(range(len(counts)), counts.values, color=colors_bar[:len(counts)])
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([c.replace("_", "\n") for c in counts.index], fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Task B: Judge Classification Distribution")
    for bar, val in zip(bars, counts.values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                    str(val), ha="center", fontsize=9)

    # Sycophancy score by emotion × strength
    ax = axes[1]
    for emo, color in PALETTE.items():
        sub = df_b[df_b["emotion"] == emo]
        if sub.empty:
            continue
        grouped = sub.groupby("strength")["is_sycophantic"].mean()
        ax.plot(grouped.index, grouped.values, "o-", color=color, label=emo, markersize=5)
    ax.set_xlabel("Steering Strength")
    ax.set_ylabel("Sycophancy Rate (judge)")
    ax.set_title("Task B: Sycophancy Rate by Condition")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(title="Emotion")

    plt.tight_layout()
    out = PLOTS_DIR / "sycophancy_distribution.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── PLOT 6: Probe heatmap (emotion probes across conditions) ─────────────────

def plot_probe_heatmap(df_a):
    probe_cols = [c for c in df_a.columns if c.startswith("probe_")]
    if not probe_cols:
        return

    pivot = df_a.groupby(["emotion", "strength"])[probe_cols].mean()
    # Show desperate and calm probes across conditions
    key_probes = ["probe_desperate", "probe_calm", "probe_angry", "probe_afraid", "probe_happy"]
    key_probes = [p for p in key_probes if p in probe_cols]

    fig, ax = plt.subplots(figsize=(10, 6))
    data = pivot[key_probes].reset_index()
    data["condition"] = data["emotion"] + " @ " + data["strength"].apply(lambda x: f"{x:+.1f}")
    data = data.set_index("condition")[key_probes]
    data.columns = [c.replace("probe_", "") for c in data.columns]

    sns.heatmap(data, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Cosine Similarity"})
    ax.set_title("Internal Emotion Probes by Steering Condition (Task A)")
    plt.tight_layout()
    out = PLOTS_DIR / "probe_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── STATISTICS ───────────────────────────────────────────────────────────────

def run_statistics(df_a, df_b):
    report = {}
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    # ── 1. Dose-response: does steering strength predict shortcut rate? ──
    print("\n--- 1. Dose-Response (Steering → Behavior) ---")
    df_a["is_shortcut"] = (df_a["judge_cls"] == "SHORTCUT").astype(int)

    # Logistic regression: shortcut ~ strength * emotion
    for emo in ["desperate", "calm"]:
        sub = df_a[df_a["emotion"] == emo].copy()
        if len(sub) < 10 or sub["is_shortcut"].nunique() < 2:
            print(f"  {emo}: insufficient variance (shortcut rate = {sub['is_shortcut'].mean():.3f})")
            continue
        X = sub[["strength"]].values
        y = sub["is_shortcut"].values
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X, y)
        auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
        # Point-biserial correlation
        r, p = stats.pointbiserialr(sub["strength"], sub["is_shortcut"])
        report[f"dose_response_a_{emo}"] = {
            "coef": float(clf.coef_[0][0]),
            "auc": float(auc),
            "correlation_r": float(r),
            "correlation_p": float(p),
            "n": len(sub),
            "shortcut_rate": float(sub["is_shortcut"].mean()),
        }
        print(f"  Task A [{emo}]: coef={clf.coef_[0][0]:.3f}, AUC={auc:.3f}, r={r:.3f}, p={p:.4f}, rate={sub['is_shortcut'].mean():.3f}")

    # Task B dose-response
    for emo in ["desperate", "calm"]:
        sub = df_b[df_b["emotion"] == emo].copy()
        if sub.empty:
            continue
        r, p = stats.pearsonr(sub["strength"], sub["syc_score_mean"])
        report[f"dose_response_b_{emo}"] = {
            "correlation_r": float(r),
            "correlation_p": float(p),
            "n": len(sub),
            "mean_syc_score": float(sub["syc_score_mean"].mean()),
        }
        print(f"  Task B [{emo}]: r={r:.3f}, p={p:.4f}, mean_score={sub['syc_score_mean'].mean():.2f}")

    # ── 2. Overall effect: unsteered vs steered ──
    print("\n--- 2. Steered vs Unsteered Comparison ---")
    for task_label, df, outcome_col in [
        ("A", df_a, "is_shortcut"),
        ("B", df_b, "syc_score_mean"),
    ]:
        unsteered = df[(df["emotion"] == "none") | (df["strength"].abs() < 1e-6)]
        steered_desp = df[(df["emotion"] == "desperate") & (df["strength"] > 0.1)]
        steered_calm = df[(df["emotion"] == "calm") & (df["strength"] > 0.1)]

        if task_label == "A":
            base_rate = unsteered[outcome_col].mean()
            desp_rate = steered_desp[outcome_col].mean() if len(steered_desp) > 0 else float("nan")
            calm_rate = steered_calm[outcome_col].mean() if len(steered_calm) > 0 else float("nan")

            # Fisher's exact test: unsteered vs desperate-steered
            if len(steered_desp) > 0:
                a = steered_desp[outcome_col].sum()
                b = len(steered_desp) - a
                c = unsteered[outcome_col].sum()
                d = len(unsteered) - c
                odds, fisher_p = stats.fisher_exact([[int(a), int(b)], [int(c), int(d)]])
                report[f"fisher_desp_{task_label}"] = {"odds_ratio": float(odds), "p": float(fisher_p)}
                print(f"  Task {task_label}: base={base_rate:.3f}, desperate_steered={desp_rate:.3f}, calm_steered={calm_rate:.3f}")
                print(f"    Fisher (desperate vs unsteered): OR={odds:.2f}, p={fisher_p:.4f}")
        else:
            base_mean = unsteered[outcome_col].mean()
            desp_mean = steered_desp[outcome_col].mean() if len(steered_desp) > 0 else float("nan")
            if len(steered_desp) > 0 and len(unsteered) > 0:
                t_stat, t_p = stats.ttest_ind(steered_desp[outcome_col], unsteered[outcome_col])
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(
                    (steered_desp[outcome_col].var() * (len(steered_desp) - 1) +
                     unsteered[outcome_col].var() * (len(unsteered) - 1)) /
                    (len(steered_desp) + len(unsteered) - 2)
                )
                cohens_d = (desp_mean - base_mean) / pooled_std if pooled_std > 0 else 0
                report[f"ttest_desp_{task_label}"] = {
                    "t": float(t_stat), "p": float(t_p), "cohens_d": float(cohens_d),
                    "base_mean": float(base_mean), "desp_mean": float(desp_mean),
                }
                print(f"  Task {task_label}: base_score={base_mean:.2f}, desperate_steered={desp_mean:.2f}")
                print(f"    t-test: t={t_stat:.3f}, p={t_p:.4f}, Cohen's d={cohens_d:.3f}")

    # ── 3. Faithfulness: V_internal vs V_text correlation ──
    print("\n--- 3. Faithfulness Correlation (V_internal vs V_text) ---")
    for task_label, df, probe_name in [("A", df_a, "desperate"), ("B", df_b, "desperate")]:
        probe_col = f"probe_{probe_name}"
        vtext_cols = [c for c in df.columns if c.startswith("vtext_")]
        df_tmp = df.copy()
        for c in vtext_cols:
            df_tmp[c] = pd.to_numeric(df_tmp[c], errors="coerce")
        df_tmp["vtext_composite"] = df_tmp[vtext_cols].mean(axis=1)

        if probe_col not in df_tmp.columns:
            continue

        valid = df_tmp.dropna(subset=[probe_col, "vtext_composite"])
        r, p = stats.pearsonr(valid[probe_col], valid["vtext_composite"])

        # Bootstrap CI
        boot_rs = []
        np.random.seed(42)
        for _ in range(10000):
            idx = np.random.randint(0, len(valid), len(valid))
            br, _ = stats.pearsonr(valid[probe_col].iloc[idx], valid["vtext_composite"].iloc[idx])
            boot_rs.append(br)
        ci_low, ci_high = np.percentile(boot_rs, [2.5, 97.5])

        report[f"faithfulness_{task_label}"] = {
            "r": float(r), "p": float(p), "n": len(valid),
            "ci_low": float(ci_low), "ci_high": float(ci_high),
        }
        print(f"  Task {task_label}: r={r:.3f} [{ci_low:.3f}, {ci_high:.3f}], p={p:.2e}, n={len(valid)}")

        # Per-dimension
        for dim_col in vtext_cols:
            dim_valid = df_tmp.dropna(subset=[probe_col, dim_col])
            if len(dim_valid) < 10:
                continue
            dim_r, dim_p = stats.pearsonr(dim_valid[probe_col], dim_valid[dim_col])
            dim_name = dim_col.replace("vtext_", "")
            print(f"    {dim_name}: r={dim_r:.3f}, p={dim_p:.4f}")

    # ── 4. Predictive comparison: V_internal vs V_text for outcome ──
    print("\n--- 4. Predictive Comparison (V_internal vs V_text → Behavior) ---")
    df_pred = df_a.copy()
    df_pred["is_shortcut"] = (df_pred["judge_cls"] == "SHORTCUT").astype(int)

    if df_pred["is_shortcut"].nunique() >= 2:
        probe_cols = [c for c in df_pred.columns if c.startswith("probe_")]
        vtext_cols = [c for c in df_pred.columns if c.startswith("vtext_")]
        for c in probe_cols + vtext_cols:
            df_pred[c] = pd.to_numeric(df_pred[c], errors="coerce").fillna(0)

        y = df_pred["is_shortcut"].values
        scaler = StandardScaler()

        for name, cols in [
            ("V_internal", probe_cols),
            ("V_text", vtext_cols),
            ("Combined", probe_cols + vtext_cols),
        ]:
            X = scaler.fit_transform(df_pred[cols].values)
            clf = LogisticRegression(max_iter=1000, random_state=42, penalty="l2", C=1.0)
            clf.fit(X, y)
            proba = clf.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, proba)
            acc = accuracy_score(y, clf.predict(X))

            # Bootstrap AUC CI
            boot_aucs = []
            np.random.seed(42)
            for _ in range(5000):
                idx = np.random.randint(0, len(y), len(y))
                if len(np.unique(y[idx])) < 2:
                    continue
                boot_aucs.append(roc_auc_score(y[idx], proba[idx]))
            ci = np.percentile(boot_aucs, [2.5, 97.5]) if boot_aucs else [0.5, 0.5]

            report[f"predictive_{name}"] = {
                "auc": float(auc), "accuracy": float(acc),
                "ci_low": float(ci[0]), "ci_high": float(ci[1]),
            }
            print(f"  {name}: AUC={auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}], acc={acc:.3f}")

    # ── 5. Kruskal-Wallis: effect of emotion on V_text dimensions ──
    print("\n--- 5. Emotion Effect on V_text (Kruskal-Wallis) ---")
    vtext_dims = ["vtext_valence", "vtext_arousal", "vtext_urgency",
                  "vtext_composure", "vtext_frustration", "vtext_hedging"]
    df_combined = pd.concat([df_a, df_b], ignore_index=True)
    for dim in vtext_dims:
        groups = [g[dim].dropna().values for _, g in df_combined.groupby("emotion") if len(g[dim].dropna()) > 5]
        if len(groups) >= 2:
            H, p = stats.kruskal(*groups)
            dim_name = dim.replace("vtext_", "")
            report[f"kruskal_{dim_name}"] = {"H": float(H), "p": float(p)}
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {dim_name}: H={H:.2f}, p={p:.4f} {sig}")

    # ── 6. Summary statistics ──
    print("\n--- 6. Summary ---")
    report["summary"] = {
        "task_a_n": len(df_a),
        "task_b_n": len(df_b),
        "task_a_shortcut_rate": float(df_a["is_shortcut"].mean()),
        "task_a_judge_cls": dict(Counter(df_a["judge_cls"])),
        "task_b_judge_cls": dict(Counter(df_b["judge_cls"])),
        "task_b_sycophancy_rate": float(df_b["is_sycophantic"].mean()),
        "emotions": sorted(df_a["emotion"].unique().tolist()),
        "strengths": sorted(df_a["strength"].unique().tolist()),
    }
    print(f"  Task A: {len(df_a)} trials, shortcut rate = {df_a['is_shortcut'].mean():.3f}")
    print(f"  Task B: {len(df_b)} trials, sycophancy rate = {df_b['is_sycophantic'].mean():.3f}")
    print(f"  Task A judge: {dict(Counter(df_a['judge_cls']))}")
    print(f"  Task B judge: {dict(Counter(df_b['judge_cls']))}")

    return report


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading judged results...")
    entries_a = load_jsonl(RESULTS_DIR / "task_a_judged.jsonl")
    entries_b = load_jsonl(RESULTS_DIR / "task_b_judged.jsonl")
    print(f"  Task A: {len(entries_a)} entries")
    print(f"  Task B: {len(entries_b)} entries")

    df_a = make_df(entries_a, "a")
    df_b = make_df(entries_b, "b")

    # Run statistics
    report = run_statistics(df_a, df_b)

    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    plot_dose_response(df_a, df_b)
    plot_judge_vs_regex(df_a, df_b)
    plot_vtext_by_condition(df_a, df_b)
    plot_faithfulness_scatter(df_a, df_b)
    plot_sycophancy_distribution(df_b)
    plot_probe_heatmap(df_a)

    # Save report
    report_path = RESULTS_DIR / "analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")
    print(f"Plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
