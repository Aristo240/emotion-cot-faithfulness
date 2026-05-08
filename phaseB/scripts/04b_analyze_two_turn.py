#!/usr/bin/env python3
"""Phase B step 4b — analyze two-turn pushback (drift predicts capitulation).

Headline tests:
  1. AUC of V_internal_t1[X], V_internal_t2[X], and Δ = V_int_t2 - V_int_t1 for
     predicting `capitulated` (judge says t2 is sycophantic and t1 was not).
  2. Does Δ add value over V_int_t2 alone? Logistic regression LR test.
  3. Boxplots of Δ by capitulation outcome.
  4. Per-category breakdown.

Outputs (idempotent — safe to run on partial data):
  results/analysis/two_turn_report.json
  results/analysis/two_turn_roc.png
  results/analysis/two_turn_drift_boxplot.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASEB = PROJECT_ROOT / "phaseB"
JUDGED = PHASEB / "results/two_turn_trials_judged.jsonl"
OUT = PHASEB / "results/analysis"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260510)


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        sys.exit(f"{p} missing — run 03b first.")
    return [json.loads(line) for line in open(p) if line.strip()]


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


def main():
    data = load_jsonl(JUDGED)
    n = len(data)
    y = np.array([1 if r.get("capitulated") else 0 for r in data])
    pos, neg = int(y.sum()), int((1 - y).sum())
    print(f"n={n}, capitulated={pos} ({pos/max(n,1):.1%}), held_firm={neg}")
    if pos == 0 or neg == 0:
        sys.exit(f"No outcome variance. capitulated={pos}, held_firm={neg}")

    EM_KEYS = ["desperate", "calm", "anxious", "frustrated", "hopeful"]
    feats = {}
    for em in EM_KEYS:
        t1 = np.array([r["emotion_probes_t1"].get(em, 0.0) for r in data])
        t2 = np.array([r["emotion_probes_t2"].get(em, 0.0) for r in data])
        feats[f"V_int_t1[{em}]"] = t1 if em != "calm" else -t1
        feats[f"V_int_t2[{em}]"] = t2 if em != "calm" else -t2
        feats[f"Δ_t2-t1[{em}]"] = (t2 - t1) if em != "calm" else (t1 - t2)

    report = {"n": n, "capitulated": pos, "held_firm": neg, "auc": {}, "ci": {}}
    for k, x in feats.items():
        a = roc_auc_score(y, x)
        lo, hi = boot_ci(y, x)
        report["auc"][k] = float(a)
        report["ci"][k] = [lo, hi]
        print(f"  {k:>26s}  AUC={a:.3f}  [{lo:.3f}, {hi:.3f}]")

    # Does Δ add value over V_int_t2?
    X_base = feats["V_int_t2[desperate]"].reshape(-1, 1)
    X_full = np.column_stack([feats["V_int_t2[desperate]"],
                              feats["Δ_t2-t1[desperate]"]])
    lr_base = LogisticRegression().fit(X_base, y)
    lr_full = LogisticRegression().fit(X_full, y)
    ll_base = -np.sum(np.log(np.clip(
        lr_base.predict_proba(X_base)[np.arange(n), y], 1e-12, 1)))
    ll_full = -np.sum(np.log(np.clip(
        lr_full.predict_proba(X_full)[np.arange(n), y], 1e-12, 1)))
    chi2 = 2 * (ll_base - ll_full)
    p_lr = 1 - stats.chi2.cdf(chi2, df=1)
    report["lr_test_delta_adds_value"] = {"chi2": float(chi2),
                                          "df": 1, "p": float(p_lr)}
    print(f"  Likelihood-ratio test (Δ adds beyond V_int_t2[desperate]): "
          f"chi2={chi2:.2f}, p={p_lr:.4f}")

    # Per-category breakdown
    cats = defaultdict(list)
    for r, yi in zip(data, y):
        cats[r.get("category", "unknown")].append((r, yi))
    per_cat = {}
    for c, items in cats.items():
        if len(items) < 30:
            continue
        ys = np.array([yi for _, yi in items])
        if ys.sum() in (0, len(ys)):
            per_cat[c] = {"n": len(ys), "events": int(ys.sum()),
                          "auc_delta_desp": None}
            continue
        xs = np.array([r["emotion_probes_t2"].get("desperate", 0.0)
                       - r["emotion_probes_t1"].get("desperate", 0.0)
                       for r, _ in items])
        per_cat[c] = {"n": len(ys), "events": int(ys.sum()),
                      "auc_delta_desp": float(roc_auc_score(ys, xs))}
    report["per_category"] = per_cat

    with open(OUT / "two_turn_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Plots
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 10})

    # ROC
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for k, color in [("V_int_t2[desperate]", "#c0392b"),
                     ("Δ_t2-t1[desperate]", "#e67e22"),
                     ("V_int_t1[desperate]", "#7f8c8d")]:
        f_p, t_p, _ = roc_curve(y, feats[k])
        ax.plot(f_p, t_p, color=color, lw=2,
                label=f"{k}  AUC={report['auc'][k]:.3f}")
    ax.plot([0, 1], [0, 1], ":", color="#bdc3c7", lw=1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title(f"Two-turn pushback: probe drift predicts capitulation\n"
                 f"n={n}, capitulated={pos}/{n}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "two_turn_roc.png")
    plt.close(fig)

    # Drift boxplot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    delta = feats["Δ_t2-t1[desperate]"]
    ax.boxplot([delta[y == 0], delta[y == 1]],
               tick_labels=[f"held firm\n(n={neg})", f"capitulated\n(n={pos})"],
               showmeans=True, meanline=True)
    ax.axhline(0, ls="--", color="gray", alpha=0.5)
    ax.set_ylabel("Δ V_internal[desperate]  (turn 2 − turn 1)")
    ax.set_title("Probe drift between turns by capitulation outcome")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "two_turn_drift_boxplot.png")
    plt.close(fig)

    print(f"\nWrote report + 2 plots to {OUT}")


if __name__ == "__main__":
    main()
