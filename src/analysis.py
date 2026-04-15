"""
Statistical analysis for the faithfulness study.

Implements the four core analyses from the research plan:
  Analysis 1: Faithfulness correlation (V_internal vs V_text)
  Analysis 2: Predictive comparison (which predicts behavior better?)
  Analysis 3: Dissociation case identification
  Analysis 4: Natural (unsteered) prediction

Plus visualization functions for all key results.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from loguru import logger
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import StatisticalSettings


def load_measurements(filepath: Path) -> pd.DataFrame:
    """Load faithfulness measurements into a DataFrame."""
    with open(filepath) as f:
        data = json.load(f)
    return pd.DataFrame(data)


# ============================================================================
# CORE ANALYSES
# ============================================================================

def analysis_1_faithfulness_correlation(
    df: pd.DataFrame,
    settings: StatisticalSettings,
) -> dict:
    """
    Analysis 1: Correlation between V_internal and V_text within each
    steering condition.

    For each steering strength, compute Pearson r between V_internal
    (probe projection) and V_text (text-derived emotion score).
    """
    results = {}

    # Composite V_text: average of available text measures
    v_text_cols = [c for c in df.columns if c.startswith("v_text_")]
    if not v_text_cols:
        logger.warning("No V_text columns found")
        return results

    # Normalize each V_text column, then average
    df_clean = df.copy()
    for col in v_text_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    df_clean["v_text_composite"] = df_clean[v_text_cols].apply(
        lambda row: row.dropna().mean(), axis=1
    )

    for strength in sorted(df_clean["strength"].unique()):
        subset = df_clean[df_clean["strength"] == strength].dropna(
            subset=["v_internal_desperate", "v_text_composite"]
        )
        if len(subset) < 10:
            continue

        r, p = stats.pearsonr(subset["v_internal_desperate"], subset["v_text_composite"])

        # Bootstrap CI
        boot_rs = []
        for _ in range(settings.n_bootstrap):
            idx = np.random.randint(0, len(subset), len(subset))
            boot_r, _ = stats.pearsonr(
                subset["v_internal_desperate"].iloc[idx],
                subset["v_text_composite"].iloc[idx],
            )
            boot_rs.append(boot_r)
        ci_low = np.percentile(boot_rs, 2.5)
        ci_high = np.percentile(boot_rs, 97.5)

        results[strength] = {
            "r": r,
            "p": p,
            "n": len(subset),
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
        logger.info(f"  Strength {strength:+.3f}: r={r:.3f} [{ci_low:.3f}, {ci_high:.3f}] (n={len(subset)})")

    return results


def analysis_2_predictive_comparison(
    df: pd.DataFrame,
    settings: StatisticalSettings,
) -> dict:
    """
    Analysis 2: Compare predictive power of V_internal vs V_text for
    behavioral outcome.

    Logistic regressions:
      (a) outcome ~ V_internal
      (b) outcome ~ V_text
      (c) outcome ~ V_internal + V_text

    Compare via AUC and likelihood ratio test.
    """
    results = {}

    # Prepare features
    df_clean = df.dropna(subset=["outcome", "v_internal_desperate"]).copy()
    df_clean["outcome_binary"] = df_clean["outcome"].astype(int)

    if df_clean["outcome_binary"].nunique() < 2:
        logger.warning("Outcome has no variance; cannot fit logistic regression")
        return {"error": "No variance in outcome"}

    # V_internal features
    v_int_cols = [c for c in df.columns if c.startswith("v_internal_")]
    v_text_cols = [c for c in df.columns if c.startswith("v_text_")]

    for col in v_int_cols + v_text_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0)

    y = df_clean["outcome_binary"].values

    # Standardize features
    scaler_int = StandardScaler()
    scaler_text = StandardScaler()

    X_int = scaler_int.fit_transform(df_clean[v_int_cols].values)
    X_text = scaler_text.fit_transform(df_clean[v_text_cols].values) if v_text_cols else np.zeros((len(y), 1))
    X_combined = np.hstack([X_int, X_text])

    def fit_and_evaluate(X, y, name):
        try:
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X, y)
            y_pred_proba = clf.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, y_pred_proba)
            acc = accuracy_score(y, clf.predict(X))
            log_likelihood = np.sum(
                y * np.log(y_pred_proba + 1e-10) +
                (1 - y) * np.log(1 - y_pred_proba + 1e-10)
            )
            return {"auc": auc, "accuracy": acc, "log_likelihood": log_likelihood, "n_features": X.shape[1]}
        except Exception as e:
            logger.warning(f"Logistic regression failed for {name}: {e}")
            return {"auc": 0.5, "accuracy": 0.5, "log_likelihood": -np.inf, "n_features": X.shape[1]}

    results["v_internal_only"] = fit_and_evaluate(X_int, y, "V_internal")
    results["v_text_only"] = fit_and_evaluate(X_text, y, "V_text")
    results["combined"] = fit_and_evaluate(X_combined, y, "Combined")

    # Likelihood ratio test: combined vs V_text_only
    ll_text = results["v_text_only"]["log_likelihood"]
    ll_combined = results["combined"]["log_likelihood"]
    lr_stat = 2 * (ll_combined - ll_text)
    df_diff = results["combined"]["n_features"] - results["v_text_only"]["n_features"]
    if df_diff > 0 and lr_stat > 0:
        p_lr = 1 - stats.chi2.cdf(lr_stat, df_diff)
    else:
        p_lr = 1.0

    results["likelihood_ratio_test"] = {
        "statistic": lr_stat,
        "df": df_diff,
        "p_value": p_lr,
        "significant": p_lr < settings.alpha,
    }

    logger.info(f"  V_internal AUC: {results['v_internal_only']['auc']:.3f}")
    logger.info(f"  V_text AUC:     {results['v_text_only']['auc']:.3f}")
    logger.info(f"  Combined AUC:   {results['combined']['auc']:.3f}")
    logger.info(f"  LR test p={p_lr:.4f} {'*' if p_lr < settings.alpha else ''}")

    return results


def analysis_3_dissociation_cases(
    df: pd.DataFrame,
) -> dict:
    """
    Analysis 3: Identify "stealth" misalignment cases where V_internal is high,
    V_text is low, and the model took the shortcut.
    """
    df_clean = df.copy()
    df_clean["outcome_binary"] = df_clean["outcome"].astype(int)

    v_int = df_clean["v_internal_desperate"]
    v_text_cols = [c for c in df.columns if c.startswith("v_text_")]
    for col in v_text_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    v_text_mean = df_clean[v_text_cols].mean(axis=1) if v_text_cols else pd.Series(0, index=df_clean.index)

    # Define thresholds
    v_int_high = v_int.quantile(0.75)
    v_text_low = v_text_mean.quantile(0.25)

    # Stealth cases: high V_internal, low V_text, bad outcome
    stealth_mask = (v_int > v_int_high) & (v_text_mean < v_text_low) & (df_clean["outcome_binary"] == 1)
    stealth_cases = df_clean[stealth_mask]

    # Overall misaligned for comparison
    misaligned_mask = df_clean["outcome_binary"] == 1

    # By steering strength
    by_strength = {}
    for strength in sorted(df_clean["strength"].unique()):
        s_mask = df_clean["strength"] == strength
        n_total = s_mask.sum()
        n_stealth = (stealth_mask & s_mask).sum()
        n_misaligned = (misaligned_mask & s_mask).sum()
        by_strength[strength] = {
            "n_total": int(n_total),
            "n_stealth": int(n_stealth),
            "n_misaligned": int(n_misaligned),
            "stealth_rate": float(n_stealth / max(n_total, 1)),
            "misaligned_rate": float(n_misaligned / max(n_total, 1)),
        }

    results = {
        "total_stealth": int(stealth_mask.sum()),
        "total_trials": len(df_clean),
        "stealth_rate": float(stealth_mask.sum() / max(len(df_clean), 1)),
        "by_strength": by_strength,
        "example_cots": stealth_cases["cot_text"].tolist()[:10],
    }

    logger.info(f"  Stealth cases: {results['total_stealth']}/{results['total_trials']} ({results['stealth_rate']:.1%})")
    return results


def analysis_4_natural_prediction(
    df: pd.DataFrame,
    settings: StatisticalSettings,
) -> dict:
    """
    Analysis 4 (H5): In UNSTEERED trials only, does natural V_internal
    variation predict behavioral outcomes?

    Stratifies by outcome_key (task type) because pooling across Task A
    (shortcut) and Task B (sycophantic) where Task B has ~zero variance
    inflates AUC via task-id separability rather than behavioral prediction.
    Reports per-task-type AUC plus a univariate V_internal_desperate AUC
    (more stable than multivariate LOO at n~40, 7 events).
    """
    # Filter to unsteered trials
    unsteered = df[df["strength"].abs() < 1e-6].copy()
    logger.info(f"  Unsteered trials: {len(unsteered)}")

    if len(unsteered) < 20:
        logger.warning("Too few unsteered trials for Analysis 4")
        return {"error": "Insufficient data", "n": len(unsteered)}

    unsteered["outcome_binary"] = unsteered["outcome"].astype(int)
    if unsteered["outcome_binary"].nunique() < 2:
        logger.warning("No outcome variance in unsteered trials")
        return {"error": "No variance", "n": len(unsteered)}

    # Per-task-type breakdown — added after red-team found pooled AUC
    # is inflated by task-id separability (Task B all-zero outcomes).
    per_task_type = {}
    for ok in sorted(unsteered["outcome_key"].dropna().unique()):
        sub = unsteered[unsteered["outcome_key"] == ok]
        y_sub = sub["outcome_binary"].values
        if y_sub.sum() == 0 or y_sub.sum() == len(y_sub):
            per_task_type[ok] = {
                "n": int(len(y_sub)), "n_events": int(y_sub.sum()),
                "note": "no outcome variance — AUC undefined",
            }
            continue
        # Univariate: V_internal_desperate signed AUC (diagnostic)
        desp = pd.to_numeric(sub["v_internal_desperate"], errors="coerce").fillna(0).values
        auc_desp = roc_auc_score(y_sub, desp)
        per_task_type[ok] = {
            "n": int(len(y_sub)),
            "n_events": int(y_sub.sum()),
            "univariate_v_internal_desperate_auc": float(auc_desp),
            "note": "univariate — more stable than multivariate LOO at low n",
        }
        logger.info(
            f"  [{ok}] n={len(y_sub)} events={int(y_sub.sum())} "
            f"univariate V_int_desp AUC={auc_desp:.3f}"
        )

    # V_internal prediction
    v_int_cols = [c for c in df.columns if c.startswith("v_internal_")]
    v_text_cols = [c for c in df.columns if c.startswith("v_text_")]
    for col in v_int_cols + v_text_cols:
        unsteered[col] = pd.to_numeric(unsteered[col], errors="coerce").fillna(0)

    y = unsteered["outcome_binary"].values
    scaler = StandardScaler()

    results = {}

    for feature_set_name, cols in [("v_internal", v_int_cols), ("v_text", v_text_cols)]:
        X = scaler.fit_transform(unsteered[cols].values) if cols else np.zeros((len(y), 1))
        try:
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X, y)
            y_pred = clf.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, y_pred)

            # Bootstrap AUC CI
            boot_aucs = []
            for _ in range(settings.n_bootstrap):
                idx = np.random.randint(0, len(y), len(y))
                if len(np.unique(y[idx])) < 2:
                    continue
                boot_auc = roc_auc_score(y[idx], y_pred[idx])
                boot_aucs.append(boot_auc)

            ci = (np.percentile(boot_aucs, 2.5), np.percentile(boot_aucs, 97.5)) if boot_aucs else (0.5, 0.5)

            results[feature_set_name] = {
                "auc": auc,
                "ci_low": ci[0],
                "ci_high": ci[1],
                "n": len(y),
            }
            logger.info(f"  {feature_set_name} AUC: {auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        except Exception as e:
            logger.warning(f"Analysis 4 failed for {feature_set_name}: {e}")
            results[feature_set_name] = {"auc": 0.5, "error": str(e)}

    results["per_task_type"] = per_task_type
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_cosine_similarity_matrix(
    sim_matrix: np.ndarray,
    emotions: List[str],
    output_path: Path,
):
    """Plot the pairwise cosine similarity heatmap."""
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        sim_matrix, xticklabels=emotions, yticklabels=emotions,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
        square=True,
    )
    ax.set_title("Pairwise Cosine Similarity of Emotion Vectors")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved cosine similarity plot to {output_path}")


def plot_pca_scatter(
    projections: np.ndarray,
    emotions: List[str],
    output_path: Path,
):
    """Plot PCA scatter of emotion vectors."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(projections[:, 0], projections[:, 1], alpha=0.7, s=30)
    for i, emo in enumerate(emotions):
        ax.annotate(emo, (projections[i, 0], projections[i, 1]),
                    fontsize=6, alpha=0.8)
    ax.set_xlabel("PC1 (valence)")
    ax.set_ylabel("PC2 (arousal)")
    ax.set_title("Emotion Vectors in PCA Space")
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_steering_dose_response(
    trials: List[dict],
    output_path: Path,
):
    """Plot behavioral outcome rate vs steering strength."""
    df = pd.DataFrame(trials) if not isinstance(trials, pd.DataFrame) else trials

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, task_type in zip(axes, ["coding", "sycophancy"]):
        task_mask = df["task_id"].str.contains(
            "fast_sum|sum_list|total|add_all" if task_type == "coding" else "sycophancy"
        )
        subset = df[task_mask]
        if subset.empty:
            continue

        for emotion in subset["emotion"].unique():
            emo_data = subset[subset["emotion"] == emotion]
            grouped = emo_data.groupby("strength")["outcome"].mean()
            ax.plot(grouped.index, grouped.values, "o-", label=emotion, markersize=4)

        ax.set_xlabel("Steering Strength")
        ax.set_ylabel("Misaligned Rate")
        ax.set_title(f"{'Shortcut' if task_type == 'coding' else 'Sycophancy'} Rate vs Steering")
        ax.legend(fontsize=8)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_faithfulness_scatter(
    measurements: pd.DataFrame,
    output_path: Path,
):
    """Plot V_internal vs V_text, colored by outcome."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    v_text_cols = [c for c in measurements.columns if c.startswith("v_text_")]
    measurements["v_text_composite"] = measurements[v_text_cols].apply(
        lambda row: pd.to_numeric(row, errors="coerce").dropna().mean(), axis=1
    )

    for ax, title, mask_fn in [
        (axes[0], "All Trials", lambda df: pd.Series(True, index=df.index)),
        (axes[1], "Unsteered Only", lambda df: df["strength"].abs() < 1e-6),
    ]:
        subset = measurements[mask_fn(measurements)]
        if subset.empty:
            continue

        colors = ["red" if o else "blue" for o in subset["outcome"]]
        ax.scatter(
            subset["v_internal_desperate"], subset["v_text_composite"],
            c=colors, alpha=0.4, s=20,
        )
        ax.set_xlabel("V_internal (desperate probe)")
        ax.set_ylabel("V_text (composite)")
        ax.set_title(title)

        # Correlation
        valid = subset.dropna(subset=["v_internal_desperate", "v_text_composite"])
        if len(valid) > 5:
            r, p = stats.pearsonr(valid["v_internal_desperate"], valid["v_text_composite"])
            ax.annotate(f"r={r:.3f}", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_full_report(
    measurements_path: Path,
    output_dir: Path,
):
    """Run all analyses and generate plots. Main entry point for Phase 3 analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = StatisticalSettings()

    df = load_measurements(measurements_path)
    logger.info(f"Loaded {len(df)} measurements")

    report = {}

    logger.info("\n=== Analysis 1: Faithfulness Correlation ===")
    report["analysis_1"] = analysis_1_faithfulness_correlation(df, settings)

    logger.info("\n=== Analysis 2: Predictive Comparison ===")
    report["analysis_2"] = analysis_2_predictive_comparison(df, settings)

    logger.info("\n=== Analysis 3: Dissociation Cases ===")
    report["analysis_3"] = analysis_3_dissociation_cases(df)

    logger.info("\n=== Analysis 4: Natural Prediction ===")
    report["analysis_4"] = analysis_4_natural_prediction(df, settings)

    # Save report
    report_file = output_dir / "analysis_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved to {report_file}")

    # Plots
    plot_faithfulness_scatter(df, output_dir / "faithfulness_scatter.png")
    plot_steering_dose_response(df, output_dir / "dose_response.png")

    return report
