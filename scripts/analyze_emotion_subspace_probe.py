#!/usr/bin/env python3
"""H_emotion_subspace: is there *any* direction in the 50-emotion subspace that
predicts SHORTCUT across mechanisms?

Three flavors:
  (a) LOGO across the 4 event-positive variants (the prereg-style test)
  (b) Within-variant 5-fold CV (upper bound — same-mechanism predictability)
  (c) Pooled-train, LOGO-eval permutation null (10000 shuffles)

Also reports which emotion directions the probe weights, to make the result
interpretable.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results/phase2/_lambda_partial/phase2/task_a_diverse_claude_judged.jsonl"
OUT = ROOT / "results/phase2/_lambda_partial/analysis_today"
OUT.mkdir(parents=True, exist_ok=True)

EVENT_POSITIVE = [
    "hardcoded_lookup_v1",
    "fake_verifier_v1",
    "silent_spec_drop_v1",
    "misleading_impl_v1",
]


def load_xy():
    trials = []
    with open(SRC) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if t["task_id"] not in EVENT_POSITIVE:
                continue
            if t["claude_classification"] == "UNCLEAR":
                continue
            trials.append(t)
    # Stable ordered list of the 50 emotion names from the first trial
    emo_names = list(trials[0]["emotion_probes"].keys())
    X = np.array([[t["emotion_probes"][k] for k in emo_names] for t in trials])
    y = np.array([1 if t["claude_classification"] == "SHORTCUT" else 0 for t in trials])
    g = np.array([t["task_id"] for t in trials])
    return X, y, g, emo_names


def fit_probe(X_train, y_train, C=1.0):
    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                             solver="liblinear")
    clf.fit(Xs, y_train)
    return clf, scaler


def auc_with_clf(clf, scaler, X, y):
    scores = clf.decision_function(scaler.transform(X))
    return roc_auc_score(y, scores), scores


def main():
    X, y, g, emo_names = load_xy()
    print(f"Loaded {len(y)} trials (LEGIT={int((y==0).sum())} SHORTCUT={int(y.sum())})")
    print(f"Variants present: {sorted(set(g))}")
    print(f"Per-variant events: {[(v, int((g==v).sum()), int(y[g==v].sum())) for v in sorted(set(g))]}\n")

    # ============================================================
    # (a) Leave-one-variant-out (LOGO)
    # ============================================================
    print("=== H_emotion_subspace LOGO (logistic regression on 50 emotion projections) ===")
    logo_aucs = {}
    coefs_per_fold = []
    for v_test in EVENT_POSITIVE:
        train_mask = g != v_test
        test_mask = g == v_test
        clf, scaler = fit_probe(X[train_mask], y[train_mask])
        auc, _ = auc_with_clf(clf, scaler, X[test_mask], y[test_mask])
        logo_aucs[v_test] = auc
        coefs_per_fold.append(clf.coef_[0])
        print(f"  Held-out={v_test:24s}  n_test={int(test_mask.sum())}  "
              f"events={int(y[test_mask].sum())}  LOGO AUC = {auc:.3f}")
    logo_mean = float(np.mean(list(logo_aucs.values())))
    print(f"  LOGO mean AUC = {logo_mean:.3f}")
    print(f"  Baseline (V_internal[desperate] LOGO mean) = 0.452")
    print(f"  Prereg threshold = 0.65\n")

    # ============================================================
    # (b) Within-variant 5-fold CV (upper bound)
    # ============================================================
    print("=== Within-variant 5-fold CV (upper bound: same mechanism) ===")
    within = {}
    for v in EVENT_POSITIVE:
        m = g == v
        if y[m].sum() < 5 or y[m].sum() == m.sum():
            continue
        Xv = X[m]; yv = y[m]
        scores = np.zeros_like(yv, dtype=float)
        for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=20260511).split(Xv, yv):
            clf, scaler = fit_probe(Xv[tr], yv[tr])
            _, s = auc_with_clf(clf, scaler, Xv[te], yv[te])
            scores[te] = s
        auc = roc_auc_score(yv, scores)
        within[v] = float(auc)
        print(f"  {v:24s} n={int(m.sum())}  events={int(yv.sum())}  within-CV AUC = {auc:.3f}")
    within_mean = float(np.mean(list(within.values())))
    print(f"  Within-variant CV mean = {within_mean:.3f}\n")

    # ============================================================
    # (c) Permutation null for LOGO
    # ============================================================
    print("=== Permutation null: shuffle labels within each variant, re-run LOGO ===")
    rng = np.random.default_rng(20260511)
    perm_means = []
    for it in range(2000):
        y_perm = y.copy()
        for v in EVENT_POSITIVE:
            mask = g == v
            yp = y_perm[mask].copy()
            rng.shuffle(yp)
            y_perm[mask] = yp
        aucs = []
        for v_test in EVENT_POSITIVE:
            tm = g != v_test
            em = g == v_test
            clf, scaler = fit_probe(X[tm], y_perm[tm])
            a, _ = auc_with_clf(clf, scaler, X[em], y_perm[em])
            aucs.append(a)
        perm_means.append(np.mean(aucs))
    perm_means = np.array(perm_means)
    p_perm = (perm_means >= logo_mean).mean()
    ci = np.percentile(perm_means, [2.5, 97.5])
    print(f"  observed LOGO mean = {logo_mean:.3f}")
    print(f"  null mean = {perm_means.mean():.3f}  null 95% range = [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"  permutation p (one-sided, observed >= null) = {p_perm:.4f}\n")

    # ============================================================
    # (d) Which emotion directions does the probe load on?
    # ============================================================
    coefs = np.stack(coefs_per_fold)  # (4 folds, 50 features)
    mean_abs = np.abs(coefs).mean(axis=0)
    mean_signed = coefs.mean(axis=0)
    order = np.argsort(-mean_abs)
    print("=== Top emotion directions used by the LOGO probe (averaged across folds) ===")
    for i in order[:15]:
        consistency = (np.sign(coefs[:, i]) == np.sign(mean_signed[i])).mean()
        print(f"  {emo_names[i]:24s}  mean_weight={mean_signed[i]:+.3f}  |w|={mean_abs[i]:.3f}  "
              f"sign_consistent_across_folds={consistency:.2f}")

    summary = {
        "n_total": int(len(y)),
        "n_events": int(y.sum()),
        "logo_aucs_per_variant": {k: float(v) for k, v in logo_aucs.items()},
        "logo_mean_auc": logo_mean,
        "v_internal_desperate_logo_mean_baseline": 0.452,
        "prereg_threshold": 0.65,
        "within_variant_cv_aucs": {k: float(v) for k, v in within.items()},
        "within_variant_cv_mean": within_mean,
        "permutation_null_mean": float(perm_means.mean()),
        "permutation_null_ci95": [float(ci[0]), float(ci[1])],
        "permutation_p_one_sided": float(p_perm),
        "top_features_by_mean_abs_weight": [
            {"emotion": emo_names[i],
             "mean_weight": float(mean_signed[i]),
             "abs_weight": float(mean_abs[i])}
            for i in order[:15]
        ],
    }
    (OUT / "h_emotion_subspace_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT/'h_emotion_subspace_summary.json'}")

    # ============================================================
    # PLOT
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    labels = list(logo_aucs.keys())
    auc_vals = [logo_aucs[v] for v in labels]
    bars = ax.bar(range(len(labels)), auc_vals, color="#2c7fb8")
    ax.axhline(0.5, color="gray", ls="--", label="chance")
    ax.axhline(0.65, color="green", ls=":", label="prereg threshold (0.65)")
    ax.axhline(logo_mean, color="#2c7fb8", ls="-", lw=1, alpha=0.6,
               label=f"LOGO mean = {logo_mean:.3f}")
    ax.axhline(0.452, color="#e74c3c", ls="-.", lw=1,
               label="V_internal[desperate] mean = 0.452")
    for i, a in enumerate(auc_vals):
        ax.text(i, a + 0.015, f"{a:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace("_v1", "") for l in labels], fontsize=8, rotation=20)
    ax.set_ylabel("AUC vs SHORTCUT")
    ax.set_ylim(0.3, 1.0)
    ax.set_title("H_emotion_subspace LOGO\n"
                 "logistic probe on 50 emotion projections, trained on 3 variants → held-out 4th")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    top_n = 12
    idx = order[:top_n]
    bars = ax.barh(range(top_n), mean_signed[idx][::-1],
                   color=["#e74c3c" if w < 0 else "#2c7fb8" for w in mean_signed[idx][::-1]])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([emo_names[i] for i in idx[::-1]], fontsize=9)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("LOGO probe mean weight (positive = pushes toward SHORTCUT)")
    ax.set_title("Top 12 emotion directions the probe uses\n"
                 "averaged across the 4 LOGO folds")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "06_emotion_subspace_logo.png", dpi=130)
    plt.close()
    print(f"Wrote plot {OUT/'06_emotion_subspace_logo.png'}")


if __name__ == "__main__":
    main()
