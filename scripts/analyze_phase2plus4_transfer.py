#!/usr/bin/env python3
"""Strengthen the fast_sum → diverse transfer finding by combining the
original Phase 2 training set with Phase 4 fast_sum trials (same task family).

Two training-set definitions:
  CONSERVATIVE: Phase 2 + Phase 4 finegrained + Phase 4 extended_unsteered
                (emotion-vector steering at various strengths only)
  FULL:         everything above + Phase 4 random_directions + text_injection
                (includes specificity-control distributions)

For each: report bootstrap CI + permutation p per diverse-test-variant +
Fisher's / Stouffer's combined p over the 3 transfer targets.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DIVERSE = ROOT / "results/phase2/_lambda_partial/phase2/task_a_diverse_claude_judged.jsonl"
OUT = ROOT / "results/phase2/_lambda_partial/analysis_today"
OUT.mkdir(parents=True, exist_ok=True)

CONSERVATIVE_FILES = [
    "results/phase2/task_a_judged.jsonl",
    "results/phase4/llama70b/finegrained_judged.jsonl",
    "results/phase4/llama70b/extended_unsteered_judged.jsonl",
]
FULL_FILES = CONSERVATIVE_FILES + [
    "results/phase4/llama70b/random_directions_judged.jsonl",
    "results/phase4/llama70b/text_injection_judged.jsonl",
]

VARIANTS_ALL = ["hardcoded_lookup_v1", "fake_verifier_v1",
                "silent_spec_drop_v1", "misleading_impl_v1"]
TRANSFER_TARGETS = ["fake_verifier_v1", "silent_spec_drop_v1", "misleading_impl_v1"]

N_BOOT = 5000
N_PERM = 1500
RNG = np.random.default_rng(20260511)


def load_jsonl_for_training(rel_paths):
    trials = []
    for rp in rel_paths:
        with open(ROOT / rp) as f:
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


def load_diverse():
    trials = []
    with open(DIVERSE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if t["task_id"] not in VARIANTS_ALL:
                continue
            if t["claude_classification"] == "UNCLEAR":
                continue
            trials.append(t)
    emo = list(trials[0]["emotion_probes"].keys())
    X = np.array([[t["emotion_probes"][k] for k in emo] for t in trials])
    y = np.array([1 if t["claude_classification"] == "SHORTCUT" else 0 for t in trials])
    g = np.array([t["task_id"] for t in trials])
    return X, y, g, emo


def align_features(X, emo, X_tr, emo_tr):
    if list(emo) == list(emo_tr):
        return X, X_tr
    common = [e for e in emo if e in emo_tr]
    ip2 = [emo_tr.index(e) for e in common]
    inew = [emo.index(e) for e in common]
    return X[:, inew], X_tr[:, ip2]


def fit(Xtr, ytr):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=2000,
                             class_weight="balanced", solver="liblinear")
    clf.fit(sc.transform(Xtr), ytr)
    return clf, sc


def evaluate_one_config(label, train_files, Xn, y, g, emo_n):
    """Run the full evaluation pipeline for one training-set definition."""
    print(f"\n{'='*78}\n   {label}\n{'='*78}")
    Xtr, ytr, emo_tr = load_jsonl_for_training(train_files)
    print(f"Training set: n={len(ytr)} SHORTCUT={int(ytr.sum())} ({100*ytr.mean():.1f}%)")
    Xn_a, Xtr_a = align_features(Xn, emo_n, Xtr, emo_tr)

    # Observed AUCs + bootstrap CI per variant
    clf, sc = fit(Xtr_a, ytr)
    scores_all = clf.decision_function(sc.transform(Xn_a))
    obs = {}
    for v in VARIANTS_ALL:
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
              f"(n={int(mask.sum())}, events={int(yv.sum())})")

    # Permutation test
    print(f"\n  Permutation test ({N_PERM} perms) ...")
    perm_aucs = {v: [] for v in VARIANTS_ALL}
    for it in range(N_PERM):
        yp = ytr.copy()
        RNG.shuffle(yp)
        clf_p, sc_p = fit(Xtr_a, yp)
        s_p = clf_p.decision_function(sc_p.transform(Xn_a))
        for v in VARIANTS_ALL:
            mask = g == v
            yv = y[mask]
            if yv.sum() < 2 or yv.sum() == mask.sum():
                continue
            perm_aucs[v].append(roc_auc_score(yv, s_p[mask]))
        if (it + 1) % 500 == 0:
            print(f"    {it+1}/{N_PERM}")

    perm_summary = {}
    for v in VARIANTS_ALL:
        if v not in obs:
            continue
        a_obs = obs[v]["auc"]
        null = np.array(perm_aucs[v])
        p_one_high = (null >= a_obs).mean()
        p_one_low = (null <= a_obs).mean()
        p_two = 2 * min(p_one_high, p_one_low)
        perm_summary[v] = {
            "obs": float(a_obs),
            "null_mean": float(null.mean()),
            "null_ci95": [float(np.percentile(null, 2.5)),
                          float(np.percentile(null, 97.5))],
            "p_one_sided_high": float(p_one_high),
            "p_two_sided": float(p_two),
        }
        print(f"  {v:24s}  obs={a_obs:.3f}  null={null.mean():.3f} "
              f"[{np.percentile(null,2.5):.3f},{np.percentile(null,97.5):.3f}]  "
              f"p(2)={p_two:.4f}")

    # Combined evidence for the 3 transfer targets (one-sided high)
    ps_one = [perm_summary[v]["p_one_sided_high"] for v in TRANSFER_TARGETS
              if v in perm_summary]
    if len(ps_one) == 3:
        chi = -2 * sum(math.log(max(p, 1.0/N_PERM)) for p in ps_one)
        p_fisher = 1 - stats.chi2.cdf(chi, df=2 * 3)
        zs = [stats.norm.isf(max(p, 1.0/N_PERM)) for p in ps_one]
        z_stouf = sum(zs) / math.sqrt(3)
        p_stouf = stats.norm.sf(z_stouf)
        print(f"\n  Per-variant one-sided p: {[f'{p:.4f}' for p in ps_one]}")
        print(f"  Fisher combined chi2={chi:.2f} df=6  p = {p_fisher:.4f}")
        print(f"  Stouffer combined z={z_stouf:.2f}  p = {p_stouf:.4f}")
    else:
        p_fisher = p_stouf = None

    return {
        "config": label,
        "train_files": train_files,
        "train_n": int(len(ytr)),
        "train_events": int(ytr.sum()),
        "per_variant": obs,
        "perm": perm_summary,
        "fisher_p": float(p_fisher) if p_fisher is not None else None,
        "stouffer_p": float(p_stouf) if p_stouf is not None else None,
    }


def main():
    Xn, y, g, emo_n = load_diverse()
    print(f"Diverse test set: n={len(y)} events={int(y.sum())}")

    r_cons = evaluate_one_config("CONSERVATIVE training (Phase 2 + Phase 4 finegrained + extended_unsteered)",
                                 CONSERVATIVE_FILES, Xn, y, g, emo_n)
    r_full = evaluate_one_config("FULL training (CONSERVATIVE + random_directions + text_injection)",
                                 FULL_FILES, Xn, y, g, emo_n)

    summary = {"conservative": r_cons, "full": r_full}
    (OUT / "phase2plus4_transfer.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT/'phase2plus4_transfer.json'}")

    # ===== Plot: side-by-side comparison =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, res in zip(axes, [r_cons, r_full]):
        labels = list(res["per_variant"].keys())
        aucs = [res["per_variant"][v]["auc"] for v in labels]
        cis_lo = [res["per_variant"][v]["ci"][0] for v in labels]
        cis_hi = [res["per_variant"][v]["ci"][1] for v in labels]
        errs = [[a - lo for a, lo in zip(aucs, cis_lo)],
                [hi - a for a, hi in zip(aucs, cis_hi)]]
        colors = ["#e74c3c" if v == "hardcoded_lookup_v1" else "#2c7fb8" for v in labels]
        bars = ax.bar(range(len(labels)), aucs, yerr=errs, capsize=5,
                      color=colors, alpha=0.85)
        for i, v in enumerate(labels):
            p_two = res["perm"][v]["p_two_sided"]
            stars = "***" if p_two < 0.001 else ("**" if p_two < 0.01 else
                                                 ("*" if p_two < 0.05 else "ns"))
            ax.text(i, aucs[i] + 0.04, f"{aucs[i]:.3f}\np={p_two:.3g} ({stars})",
                    ha="center", fontsize=8)
        ax.axhline(0.5, color="gray", ls="--", label="chance")
        ax.axhline(0.65, color="green", ls=":", label="prereg threshold (0.65)")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([v.replace("_v1", "").replace("_", "\n") for v in labels],
                           fontsize=8.5)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("AUC")
        ax.set_title(f"{res['config'].split(' (')[0]}\n"
                     f"train n={res['train_n']}, events={res['train_events']}  |  "
                     f"Fisher p={res['fisher_p']:.4f}  Stouffer p={res['stouffer_p']:.4f}",
                     fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "09_phase2plus4_transfer.png", dpi=130)
    plt.close()
    print(f"Wrote plot {OUT/'09_phase2plus4_transfer.png'}")


if __name__ == "__main__":
    main()
