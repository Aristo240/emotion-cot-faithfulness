#!/usr/bin/env python3
"""
H5 held-out evaluation — addresses red-team finding F2.

Headline rigor-report number (univariate AUC = 0.901 [0.79, 0.98] on n=40,
events=7, Task A unsteered) is computed in-distribution: the V_internal
direction is frozen from Phase 1 (zero-shot, no fitting on Phase 3 data),
but evaluation pools all 4 fast_sum variants together. Because those
variants share the same shortcut mechanism, pooled AUC may over-state
cross-task generalization. Reviewer-grade evaluation needs:

  (1) Pooled univariate AUC — current headline, kept for comparison.
  (2) Leave-one-task-out AUC — frozen Phase-1 direction, evaluate on
      held-out task variant. Direct test of cross-prompt generalization.
  (3) Permutation-test p-value — shuffle outcome labels (preserving event
      count) and recompute AUC; report fraction >= observed AUC.
  (4) Stratified bootstrap CI — resample positives and negatives
      separately (preserves event-count noise structure).

This script is read-only with respect to upstream measurements; it loads
results/phase3/llama70b/faithfulness_measurements.json. After Phase 4 (B)
ext_unsteered.jsonl lands, re-run Phase 3 first to refresh measurements,
then re-run this script — the protocol is locked in docs/preregistration.md.

Usage:
    python scripts/h5_holdout.py --model llama-70b
    python scripts/h5_holdout.py --model llama-70b --n-perm 10000 --n-boot 5000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR, MODELS

RNG_SEED = 20260415  # locked seed; do not change per preregistration


def load_unsteered_taskA(measurements_path: Path) -> pd.DataFrame:
    with open(measurements_path) as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    df = df[df["outcome_key"] == "shortcut"].copy()
    df = df[df["strength"] == 0.0].copy()  # unsteered
    df["y"] = df["outcome"].astype(int)
    df["v_int_desp"] = pd.to_numeric(df["v_internal_desperate"], errors="coerce").fillna(0.0)
    df = df.reset_index(drop=True)
    return df


def auc_or_nan(y, x):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, x))


def pooled_univariate(df: pd.DataFrame) -> dict:
    return {
        "n": int(len(df)),
        "events": int(df["y"].sum()),
        "auc": auc_or_nan(df["y"].values, df["v_int_desp"].values),
    }


# Preregistered: refuse to report a confirmatory LOGO mean if fewer
# than MIN_TASKS_WITH_EVENTS task variants produced ≥1 event. The
# threshold scales with the total suite size (pre/post diverse variants).
# 4 original variants → require 3. 9 total (4 original + 5 diverse) →
# require 6. Script auto-computes this from the observed n_total_tasks.
def min_tasks_required(n_total: int) -> int:
    if n_total <= 4:
        return 3
    # ≥5: require two-thirds (ceiling)
    return int(np.ceil(2 * n_total / 3))


def leave_one_task_out(df: pd.DataFrame) -> dict:
    """Per-task AUC of the frozen V_internal direction. Note: because the
    direction is fixed (not refit per held-out task), this is *per-task
    evaluation* of the same predictor — it's not cross-validation in the
    standard sense. The relevant question is whether the same fixed
    direction discriminates within each task individually.

    LOGO mean is reported only if at least MIN_TASKS_WITH_EVENTS_FOR_LOGO
    task variants have outcome variance. Otherwise the mean is meaningless
    (e.g., averaging over 2 tasks of 4 hides the fact that 50% of the
    task suite produced no signal)."""
    per_task = {}
    for tid in sorted(df["task_id"].unique()):
        sub = df[df["task_id"] == tid]
        ys = sub["y"].values
        xs = sub["v_int_desp"].values
        per_task[tid] = {
            "n": int(len(sub)),
            "events": int(ys.sum()),
            "auc": auc_or_nan(ys, xs),
        }
    aucs_with_var = [v["auc"] for v in per_task.values() if not np.isnan(v["auc"])]
    n_with_events = len(aucs_with_var)
    n_total = len(per_task)
    min_req = min_tasks_required(n_total)
    if n_with_events >= min_req:
        mean_auc = float(np.mean(aucs_with_var))
        median_auc = float(np.median(aucs_with_var))
        logo_meaningful = True
    else:
        mean_auc = float("nan")
        median_auc = float("nan")
        logo_meaningful = False
    return {
        "per_task": per_task,
        "mean_auc_across_tasks": mean_auc,
        "median_auc_across_tasks": median_auc,
        "n_tasks_with_variance": n_with_events,
        "n_tasks_total": n_total,
        "min_tasks_required": min_req,
        "logo_meaningful": logo_meaningful,
        "_note": (
            f"LOGO mean reported only when >= {min_req} of {n_total} "
            f"task variants have outcome variance. Currently "
            f"{n_with_events}/{n_total}."
        ),
    }


def permutation_test(df: pd.DataFrame, n_perm: int, rng: np.random.Generator) -> dict:
    y = df["y"].values
    x = df["v_int_desp"].values
    obs = auc_or_nan(y, x)
    if np.isnan(obs):
        return {"observed_auc": obs, "p_value": float("nan"), "n_perm": n_perm}
    null_aucs = np.empty(n_perm)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        null_aucs[i] = roc_auc_score(y_perm, x)
    # one-sided: how often the null AUC matches or beats observed
    p = float((np.sum(null_aucs >= obs) + 1) / (n_perm + 1))
    return {
        "observed_auc": float(obs),
        "p_value": p,
        "null_mean": float(null_aucs.mean()),
        "null_q95": float(np.quantile(null_aucs, 0.95)),
        "n_perm": int(n_perm),
    }


def stratified_bootstrap(df: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> dict:
    """Resample positives and negatives separately to preserve the event-count
    structure. With n=40 / 7 events, naive bootstrap can produce samples with
    0 events (degenerate AUC); stratified sampling avoids this and gives an
    honest CI conditional on the observed prevalence."""
    pos = df[df["y"] == 1]
    neg = df[df["y"] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return {"auc_mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    aucs = np.empty(n_boot)
    for i in range(n_boot):
        p_idx = rng.integers(0, len(pos), len(pos))
        n_idx = rng.integers(0, len(neg), len(neg))
        sub = pd.concat([pos.iloc[p_idx], neg.iloc[n_idx]])
        aucs[i] = auc_or_nan(sub["y"].values, sub["v_int_desp"].values)
    aucs = aucs[~np.isnan(aucs)]
    return {
        "auc_mean": float(aucs.mean()),
        "ci_low": float(np.quantile(aucs, 0.025)),
        "ci_high": float(np.quantile(aucs, 0.975)),
        "n_boot": int(len(aucs)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    short = MODELS[args.model].short_name
    in_path = RESULTS_DIR / "phase3" / short / "faithfulness_measurements.json"
    out_dir = RESULTS_DIR / "phase3" / short
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "h5_holdout_report.json"

    df = load_unsteered_taskA(in_path)
    rng = np.random.default_rng(RNG_SEED)

    pooled = pooled_univariate(df)
    logo = leave_one_task_out(df)
    perm = permutation_test(df, args.n_perm, rng)
    boot = stratified_bootstrap(df, args.n_boot, rng)

    # Decision rule (preregistered): LOGO must be meaningful AND mean >= 0.70
    # AND permutation p < 0.01. If LOGO is not meaningful (too few tasks with
    # variance), we explicitly DECLINE to make a confirmatory claim regardless
    # of how good pooled AUC looks.
    decision = {
        "rule": (
            "logo_meaningful AND logo_mean_auc >= 0.70 AND permutation_p < 0.01"
        ),
        "logo_meaningful": logo["logo_meaningful"],
        "logo_mean_auc_pass": (
            logo["logo_meaningful"]
            and not np.isnan(logo["mean_auc_across_tasks"])
            and logo["mean_auc_across_tasks"] >= 0.70
        ),
        "permutation_pass": perm["p_value"] < 0.01,
        "all_pass": False,
        "verdict": None,
    }
    decision["all_pass"] = bool(
        decision["logo_meaningful"]
        and decision["logo_mean_auc_pass"]
        and decision["permutation_pass"]
    )
    if not decision["logo_meaningful"]:
        decision["verdict"] = (
            "INSUFFICIENT-DATA: cannot make confirmatory H5 claim. "
            "Re-run after Phase 4 (B) extended_unsteered.jsonl lands and "
            "after diverse Task A variants have been collected."
        )
    elif decision["all_pass"]:
        decision["verdict"] = "H5-SUPPORTED (pre-registered rule met)"
    else:
        decision["verdict"] = "H5-NOT-SUPPORTED (pre-registered rule failed)"

    report = {
        "rng_seed": RNG_SEED,
        "input": str(in_path),
        "n_unsteered_taskA": int(len(df)),
        "events_unsteered_taskA": int(df["y"].sum()),
        "(1)_pooled_univariate": pooled,
        "(2)_leave_one_task_out": logo,
        "(3)_permutation_test": perm,
        "(4)_stratified_bootstrap_ci": boot,
        "(5)_decision": decision,
        "preregistered_protocol": "docs/preregistration.md",
        "_note": (
            "Pooled AUC and stratified bootstrap CI are reported as "
            "exploratory — only the (5) decision is confirmatory."
        ),
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Human summary
    print("=" * 72)
    print("H5 HELD-OUT REPORT")
    print("=" * 72)
    print(f"n={pooled['n']}, events={pooled['events']}")
    print(f"(1) Pooled univariate AUC          = {pooled['auc']:.3f}  [exploratory]")
    if logo["logo_meaningful"]:
        print(f"(2) LOGO mean AUC across tasks     = "
              f"{logo['mean_auc_across_tasks']:.3f}  "
              f"({logo['n_tasks_with_variance']}/{logo['n_tasks_total']} tasks)")
    else:
        print(f"(2) LOGO mean AUC                  = N/A "
              f"(only {logo['n_tasks_with_variance']}/{logo['n_tasks_total']} "
              f"tasks have events; need >= {logo['min_tasks_required']})")
    for tid, v in logo["per_task"].items():
        print(f"      {tid}: AUC={v['auc']}, n={v['n']}, events={v['events']}")
    print(f"(3) Permutation p (n={args.n_perm})    = {perm['p_value']:.4g}")
    print(f"(4) Stratified bootstrap 95% CI    = "
          f"[{boot['ci_low']:.3f}, {boot['ci_high']:.3f}], mean {boot['auc_mean']:.3f}  "
          f"[exploratory]")
    print(f"(5) DECISION: {decision['verdict']}")
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
