#!/usr/bin/env python3
"""Are the per-mechanism shortcut directions in the emotion subspace aligned,
or genuinely different?

(1) Train one logistic probe per mechanism (within-variant) → 4 weight vectors
    w_v in R^50. Report pairwise cosine similarity. High → shared direction.
    Low → each mechanism uses a different combination of emotion features.

(2) Train one probe on the ORIGINAL Phase 2 task_a_judged data (the fast_sum
    family where V_internal[desperate] AUC was 0.84). Apply this "Phase 2
    direction" to each new diverse-variant trial. Does the Phase 2 direction
    transfer?

(3) Within-variant LOGO with z-scored features per mechanism (sanity: does the
    AUC story change if we equalize feature scales per mechanism?).
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
DIVERSE = ROOT / "results/phase2/_lambda_partial/phase2/task_a_diverse_claude_judged.jsonl"
PHASE2 = ROOT / "results/phase2/task_a_judged.jsonl"
OUT = ROOT / "results/phase2/_lambda_partial/analysis_today"
OUT.mkdir(parents=True, exist_ok=True)

EVENT_POSITIVE = [
    "hardcoded_lookup_v1",
    "fake_verifier_v1",
    "silent_spec_drop_v1",
    "misleading_impl_v1",
]


def load_diverse():
    trials = []
    with open(DIVERSE) as f:
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
    emo = list(trials[0]["emotion_probes"].keys())
    X = np.array([[t["emotion_probes"][k] for k in emo] for t in trials])
    y = np.array([1 if t["claude_classification"] == "SHORTCUT" else 0 for t in trials])
    g = np.array([t["task_id"] for t in trials])
    return X, y, g, emo


def load_phase2_original():
    """Original Phase 2 task_a_judged.jsonl — 4 fast_sum variants, 520 trials."""
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
    if not trials:
        return None, None, None
    emo = list(trials[0][0]["emotion_probes"].keys())
    X = np.array([[t["emotion_probes"][k] for k in emo] for t, _ in trials])
    y = np.array([1 if jc == "SHORTCUT" else 0 for _, jc in trials])
    return X, y, emo


def fit(Xtr, ytr, C=1.0):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                             solver="liblinear")
    clf.fit(sc.transform(Xtr), ytr)
    return clf, sc


def cos(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def main():
    X, y, g, emo = load_diverse()
    print(f"Diverse trials: n={len(y)}, events={y.sum()}")
    print(f"Variants present: {sorted(set(g))}")

    # ============================================================
    # (1) Per-mechanism within-variant probes — get one weight vec per variant
    # ============================================================
    print("\n=== (1) Per-mechanism probe directions ===")
    per_mech_w = {}
    per_mech_auc = {}
    for v in EVENT_POSITIVE:
        mask = g == v
        if y[mask].sum() < 5 or y[mask].sum() == mask.sum():
            print(f"  {v}: skipped — not enough events")
            continue
        Xv = X[mask]; yv = y[mask]
        # within-CV scores for the AUC report
        scores = np.zeros_like(yv, dtype=float)
        for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=20260511).split(Xv, yv):
            clf, sc = fit(Xv[tr], yv[tr])
            scores[te] = clf.decision_function(sc.transform(Xv[te]))
        per_mech_auc[v] = float(roc_auc_score(yv, scores))
        # Full-fit on all of this variant to get a single representative w
        clf, sc = fit(Xv, yv)
        per_mech_w[v] = clf.coef_[0]
        print(f"  {v:24s}  within-CV AUC = {per_mech_auc[v]:.3f}")

    # Pairwise cosine similarity matrix
    print("\n=== Pairwise cosine similarity between per-mechanism probe weight vectors ===")
    keys = list(per_mech_w.keys())
    M = np.zeros((len(keys), len(keys)))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            M[i, j] = cos(per_mech_w[ki], per_mech_w[kj])
    # print
    header = " " * 24 + "  ".join(f"{k.replace('_v1',''):>15s}" for k in keys)
    print(header)
    for i, ki in enumerate(keys):
        row = "  ".join(f"{M[i,j]:+.3f}".rjust(15) for j in range(len(keys)))
        print(f"  {ki:24s}{row}")
    off_diag = [M[i, j] for i in range(len(keys)) for j in range(len(keys)) if i != j]
    print(f"\n  Mean off-diagonal cosine = {np.mean(off_diag):+.3f}")
    print(f"  Range = [{min(off_diag):+.3f}, {max(off_diag):+.3f}]")

    # ============================================================
    # (2) Train probe on ORIGINAL Phase 2 fast_sum data, evaluate on diverse
    # ============================================================
    print("\n=== (2) Train on original Phase 2 fast_sum, evaluate on new diverse variants ===")
    Xp2, yp2, emo_p2 = load_phase2_original()
    if Xp2 is None:
        print("  Original Phase 2 task_a_judged.jsonl not loadable; skipping.")
        phase2_transfer = None
    else:
        print(f"  Phase 2 training set: n={len(yp2)} events={yp2.sum()}")
        if list(emo_p2) != list(emo):
            common = [e for e in emo if e in emo_p2]
            ip2 = [emo_p2.index(e) for e in common]
            inew = [emo.index(e) for e in common]
            Xp2_a = Xp2[:, ip2]; Xnew_a = X[:, inew]
            print(f"  Aligning {len(common)} common emotion features")
        else:
            Xp2_a = Xp2; Xnew_a = X
        clf, sc = fit(Xp2_a, yp2)
        # AUC on each diverse variant separately
        phase2_transfer = {}
        for v in EVENT_POSITIVE:
            mask = g == v
            yv = y[mask]
            if yv.sum() < 3 or yv.sum() == mask.sum():
                continue
            s = clf.decision_function(sc.transform(Xnew_a[mask]))
            phase2_transfer[v] = float(roc_auc_score(yv, s))
            print(f"  Apply Phase 2 probe to {v:24s}  AUC = {phase2_transfer[v]:.3f}  (n={mask.sum()}, events={int(yv.sum())})")
        # Also pooled
        all_scores = clf.decision_function(sc.transform(Xnew_a))
        print(f"  Pooled across diverse:    AUC = {roc_auc_score(y, all_scores):.3f}")
        # And: cosine between Phase 2 probe direction and each per-mechanism direction
        wp2 = clf.coef_[0]
        if len(wp2) != len(next(iter(per_mech_w.values()))):
            # rebuild per-mech w using only common features for fair cosine
            print("  (cosine comparison skipped: dim mismatch)")
        else:
            print("\n  Cosine similarity: Phase 2 direction vs each per-mechanism direction:")
            for v in keys:
                print(f"    Phase 2 ↔ {v:24s}  cos = {cos(wp2, per_mech_w[v]):+.3f}")

    # ============================================================
    # (3) PCA: how much variance in shortcut probability comes from 1 direction?
    # ============================================================
    # Compute pseudo-direction from full pooled fit
    print("\n=== (3) Effective dimensionality of the shortcut direction ===")
    clf_pool, sc_pool = fit(X, y)
    w_pool = clf_pool.coef_[0]
    # Project each per-mech weight onto pool-direction
    print(f"  Cosine between per-mechanism w and pooled w:")
    for v in keys:
        print(f"    {v:24s}  cos = {cos(per_mech_w[v], w_pool):+.3f}")

    # ============================================================
    # SAVE
    # ============================================================
    summary = {
        "per_mech_within_cv_auc": per_mech_auc,
        "cosine_matrix": {ki: {kj: float(M[i, j]) for j, kj in enumerate(keys)}
                          for i, ki in enumerate(keys)},
        "mean_off_diag_cosine": float(np.mean(off_diag)),
        "phase2_transfer_aucs": phase2_transfer,
        "cosine_pool_vs_per_mech": {v: float(cos(per_mech_w[v], w_pool)) for v in keys},
    }
    (OUT / "per_mech_directions_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT/'per_mech_directions_summary.json'}")

    # ============================================================
    # PLOT
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Heatmap of cosines
    ax = axes[0]
    im = ax.imshow(M, cmap="RdBu", vmin=-1, vmax=1)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center",
                    color="white" if abs(M[i, j]) > 0.5 else "black", fontsize=10)
    ax.set_xticks(range(len(keys))); ax.set_yticks(range(len(keys)))
    ax.set_xticklabels([k.replace("_v1", "") for k in keys], rotation=25, fontsize=9)
    ax.set_yticklabels([k.replace("_v1", "") for k in keys], fontsize=9)
    ax.set_title("Cosine similarity between per-mechanism\nlogistic-probe weight vectors (R^50)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Phase 2 transfer bar
    ax = axes[1]
    if phase2_transfer:
        labels = list(phase2_transfer.keys())
        vals = [phase2_transfer[v] for v in labels]
        bars = ax.bar(range(len(labels)), vals, color="#9b59b6")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
        ax.axhline(0.5, color="gray", ls="--", label="chance")
        ax.axhline(0.65, color="green", ls=":", label="prereg threshold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace("_v1", "") for l in labels], rotation=20, fontsize=8)
        ax.set_ylim(0.3, 1.0)
        ax.set_ylabel("AUC vs SHORTCUT")
        ax.set_title("Probe trained on original Phase 2 (fast_sum family)\napplied to each new diverse variant")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "07_per_mech_directions.png", dpi=130)
    plt.close()
    print(f"Wrote plot {OUT/'07_per_mech_directions.png'}")


if __name__ == "__main__":
    main()
