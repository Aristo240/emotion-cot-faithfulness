#!/usr/bin/env python3
"""Solidify the Phase 2 → diverse-variant transfer finding with:

(a) Bootstrap 95% CI per variant + pooled (over non-control variants)
(b) Permutation test: shuffle Phase 2 SHORTCUT labels, refit logistic regression,
    evaluate AUC on each diverse variant. 2000 perms. One-sided p-value.

The 3 "non-control" variants we pool for the headline claim are
fake_verifier_v1, silent_spec_drop_v1, misleading_impl_v1.
hardcoded_lookup_v1 is reported separately as the dissociation control.
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
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DIVERSE = ROOT / "results/phase2/_lambda_partial/phase2/task_a_diverse_claude_judged.jsonl"
PHASE2 = ROOT / "results/phase2/task_a_judged.jsonl"
OUT = ROOT / "results/phase2/_lambda_partial/analysis_today"
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = ["hardcoded_lookup_v1", "fake_verifier_v1",
            "silent_spec_drop_v1", "misleading_impl_v1"]
TRANSFER_TARGETS = ["fake_verifier_v1", "silent_spec_drop_v1", "misleading_impl_v1"]
CONTROL = "hardcoded_lookup_v1"

N_BOOT = 5000
N_PERM = 2000
RNG = np.random.default_rng(20260511)


def load_diverse():
    trials = []
    with open(DIVERSE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if t["task_id"] not in VARIANTS:
                continue
            if t["claude_classification"] == "UNCLEAR":
                continue
            trials.append(t)
    emo = list(trials[0]["emotion_probes"].keys())
    X = np.array([[t["emotion_probes"][k] for k in emo] for t in trials])
    y = np.array([1 if t["claude_classification"] == "SHORTCUT" else 0 for t in trials])
    g = np.array([t["task_id"] for t in trials])
    return X, y, g, emo


def load_phase2():
    trials = []
    with open(PHASE2) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            jc = t.get("judge_classification") or t.get("classification")
            if jc not in ("SHORTCUT", "LEGITIMATE"):
                continue
            if not t.get("emotion_probes"):
                continue
            trials.append((t, jc))
    emo = list(trials[0][0]["emotion_probes"].keys())
    X = np.array([[t["emotion_probes"][k] for k in emo] for t, _ in trials])
    y = np.array([1 if jc == "SHORTCUT" else 0 for _, jc in trials])
    return X, y, emo


def fit(Xtr, ytr):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                             solver="liblinear")
    clf.fit(sc.transform(Xtr), ytr)
    return clf, sc


def align(X, y, emo, X_p2, emo_p2):
    """Align column ordering between two feature matrices."""
    if list(emo) == list(emo_p2):
        return X, X_p2
    common = [e for e in emo if e in emo_p2]
    ip2 = [emo_p2.index(e) for e in common]
    inew = [emo.index(e) for e in common]
    return X[:, inew], X_p2[:, ip2]


def main():
    X, y, g, emo = load_diverse()
    Xp2, yp2, emo_p2 = load_phase2()
    Xn, Xp2 = align(X, y, emo, Xp2, emo_p2)
    print(f"Phase 2 training: n={len(yp2)} events={int(yp2.sum())}")
    print(f"Diverse: n={len(y)} events={int(y.sum())}")
    print(f"Feature dim after alignment: {Xn.shape[1]}")

    # ============================================================
    # (a) Observed AUCs + bootstrap CIs per variant
    # ============================================================
    clf, sc = fit(Xp2, yp2)
    scores_all = clf.decision_function(sc.transform(Xn))
    print("\n=== Phase 2 → diverse: observed AUCs + bootstrap 95% CI per variant ===")
    obs = {}
    for v in VARIANTS:
        mask = g == v
        yv = y[mask]; sv = scores_all[mask]
        if yv.sum() < 3 or yv.sum() == mask.sum():
            continue
        a = roc_auc_score(yv, sv)
        boots = []
        for _ in range(N_BOOT):
            idx = RNG.integers(0, len(yv), len(yv))
            yb = yv[idx]
            if yb.sum() < 2 or yb.sum() == len(yb):
                continue
            boots.append(roc_auc_score(yb, sv[idx]))
        ci = np.percentile(boots, [2.5, 97.5])
        obs[v] = {"auc": float(a), "ci": [float(ci[0]), float(ci[1])],
                  "n": int(mask.sum()), "events": int(yv.sum())}
        print(f"  {v:24s}  AUC = {a:.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]  "
              f"(n={mask.sum()}, events={int(yv.sum())})")

    # Pooled across the 3 transfer targets (the headline claim)
    mask3 = np.isin(g, TRANSFER_TARGETS)
    y3 = y[mask3]; s3 = scores_all[mask3]
    a3 = roc_auc_score(y3, s3)
    boots3 = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, len(y3), len(y3))
        if y3[idx].sum() < 2 or y3[idx].sum() == len(y3[idx]):
            continue
        boots3.append(roc_auc_score(y3[idx], s3[idx]))
    ci3 = np.percentile(boots3, [2.5, 97.5])
    print(f"\n  POOLED across 3 transfer targets (excluding hardcoded_lookup):")
    print(f"    n={mask3.sum()} events={int(y3.sum())}  AUC = {a3:.3f}  "
          f"95% CI [{ci3[0]:.3f}, {ci3[1]:.3f}]")
    obs["pooled_3_transfer"] = {"auc": float(a3),
                                 "ci": [float(ci3[0]), float(ci3[1])],
                                 "n": int(mask3.sum()),
                                 "events": int(y3.sum())}

    # ============================================================
    # (b) Permutation test: shuffle Phase 2 labels, refit, evaluate
    # ============================================================
    print(f"\n=== Permutation test ({N_PERM} perms): shuffle Phase 2 labels, refit, evaluate ===")
    print("  (This tests whether the cross-mechanism transfer could arise by chance)")
    perm_aucs = {v: [] for v in VARIANTS}
    perm_pooled = []
    yp2_copy = yp2.copy()
    for it in range(N_PERM):
        yp_perm = yp2_copy.copy()
        RNG.shuffle(yp_perm)
        clf_p, sc_p = fit(Xp2, yp_perm)
        s_p = clf_p.decision_function(sc.transform(Xn))  # same scaler? actually use sc_p
        s_p = clf_p.decision_function(sc_p.transform(Xn))
        for v in VARIANTS:
            mask = g == v
            yv = y[mask]
            if yv.sum() < 2 or yv.sum() == mask.sum():
                continue
            perm_aucs[v].append(roc_auc_score(yv, s_p[mask]))
        perm_pooled.append(roc_auc_score(y[mask3], s_p[mask3]))
        if (it + 1) % 500 == 0:
            print(f"    {it+1}/{N_PERM} perms done")

    print()
    perm_summary = {}
    for v in VARIANTS:
        if v not in obs:
            continue
        a_obs = obs[v]["auc"]
        null = np.array(perm_aucs[v])
        # Two-sided permutation p: 2 * min(P(null<=obs), P(null>=obs))
        p_two = 2 * min((null <= a_obs).mean(), (null >= a_obs).mean())
        null_mean = float(null.mean()); null_ci = np.percentile(null, [2.5, 97.5])
        perm_summary[v] = {"observed": float(a_obs),
                           "null_mean": null_mean,
                           "null_ci95": [float(null_ci[0]), float(null_ci[1])],
                           "p_two_sided": float(p_two)}
        print(f"  {v:24s}  obs={a_obs:.3f}  null={null_mean:.3f} "
              f"[{null_ci[0]:.3f},{null_ci[1]:.3f}]  p(2-sided)={p_two:.4f}")

    pn = np.array(perm_pooled)
    p_pool_one = (pn >= a3).mean()
    p_pool_two = 2 * min((pn <= a3).mean(), (pn >= a3).mean())
    print(f"\n  POOLED 3-target  obs={a3:.3f}  null_mean={pn.mean():.3f}  "
          f"p(1-sided)={p_pool_one:.4f}  p(2-sided)={p_pool_two:.4f}")
    perm_summary["pooled_3_transfer"] = {
        "observed": float(a3), "null_mean": float(pn.mean()),
        "null_ci95": [float(np.percentile(pn, 2.5)), float(np.percentile(pn, 97.5))],
        "p_one_sided": float(p_pool_one), "p_two_sided": float(p_pool_two),
    }

    summary = {"observed": obs, "permutation": perm_summary,
               "n_boot": N_BOOT, "n_perm": N_PERM}
    (OUT / "phase2_transfer_stats.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT/'phase2_transfer_stats.json'}")

    # ============================================================
    # PLOT: AUC with CI bars + null distribution markers
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    show = list(obs.keys())
    xs = np.arange(len(show))
    aucs = [obs[v]["auc"] for v in show]
    ci_lo = [obs[v]["ci"][0] for v in show]
    ci_hi = [obs[v]["ci"][1] for v in show]
    errs = [[a - lo for a, lo in zip(aucs, ci_lo)],
            [hi - a for a, hi in zip(aucs, ci_hi)]]
    colors = ["#e74c3c" if v == CONTROL else
              ("#9b59b6" if v == "pooled_3_transfer" else "#2c7fb8") for v in show]
    ax.bar(xs, aucs, yerr=errs, capsize=5, color=colors, alpha=0.85)
    for x, v, a in zip(xs, show, aucs):
        p_lab = ""
        if v in perm_summary:
            p_lab = f"\np={perm_summary[v].get('p_two_sided', perm_summary[v].get('p_one_sided')):.3g}"
        ax.text(x, a + 0.04, f"{a:.3f}{p_lab}", ha="center", fontsize=8)
    ax.axhline(0.5, color="gray", ls="--", label="chance")
    ax.axhline(0.65, color="green", ls=":", label="prereg threshold (0.65)")
    ax.set_xticks(xs)
    ax.set_xticklabels([v.replace("_v1", "").replace("_", "_\n").replace("pooled_3_transfer", "POOLED\n3 transfer")
                        for v in show], fontsize=8.5)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.1, 1.0)
    ax.set_title("Phase 2 (fast_sum) shortcut probe transferred to diverse variants\n"
                 "Bars = observed AUC, error bars = 95% bootstrap CI, p = permutation test (Phase 2 labels shuffled)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "08_phase2_transfer_with_stats.png", dpi=130)
    plt.close()
    print(f"Wrote plot {OUT/'08_phase2_transfer_with_stats.png'}")


if __name__ == "__main__":
    main()
