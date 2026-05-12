#!/usr/bin/env python3
"""V_internal vs V_text divergence analysis (per Gemini's suggestion).

The paired bootstrap test showed ΔAUC ≈ 0 between V_internal[desperate] and
V_text[desperate-proxy]. But "no AUC gap" can mean two very different things:
  (a) The two predictors agree on essentially every trial (same shortcut events
      caught and missed) → the negative result is strong: no faithfulness gap,
      surface monitoring sees the same signal as internal probing.
  (b) The two predictors disagree per trial but produce equal AUC by accident
      (their errors cancel in aggregate) → there's complementary information
      that a combined predictor could exploit.

This script distinguishes (a) from (b) on three datasets:
  D1: Phase 4 extended_unsteered_judged  (the strongest H5 dataset, AUC 0.84
                                          for V_internal[desperate])
  D2: Phase 4 finegrained_judged + Phase 2 task_a_judged (fast_sum family)
  D3: phase2_diverse_claude_judged       (new cross-mechanism dataset)

For each:
  * Pearson and Spearman correlation between V_internal[desperate] and V_text
  * 4x4 quartile-by-quartile hack-rate heatmap
  * Disagreement-cell hack rates (high-V_internal/low-V_text, low/high)
  * Conditional AUC: V_internal AUC restricted to trials where V_text is low
    (the trials where surface monitoring would have said "calm"); and the
    symmetric V_text AUC restricted to trials where V_internal is low
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DIVERSE = ROOT / "results/phase2/_lambda_partial/phase2/task_a_diverse_claude_judged.jsonl"
PHASE2 = ROOT / "results/phase2/task_a_judged.jsonl"
P4_EXT = ROOT / "results/phase4/llama70b/extended_unsteered_judged.jsonl"
P4_FG = ROOT / "results/phase4/llama70b/finegrained_judged.jsonl"
OUT = ROOT / "results/phase2/_lambda_partial/analysis_today"
OUT.mkdir(parents=True, exist_ok=True)


def vtext_desperate(vt: dict) -> float:
    if not vt:
        return np.nan
    return (vt.get("urgency", 3) - vt.get("composure", 3) + vt.get("frustration", 3)) / 3.0


def load(path: Path, judge_field: str = "judge_classification") -> list[dict]:
    trials = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            jc = t.get(judge_field) or t.get("judge_classification") or t.get("classification")
            if jc not in ("SHORTCUT", "LEGITIMATE"):
                continue
            if not t.get("emotion_probes") or not t.get("vtext_ratings"):
                continue
            t["_y"] = 1 if jc == "SHORTCUT" else 0
            t["_vint"] = t["emotion_probes"]["desperate"]
            t["_vint_neg_calm"] = -t["emotion_probes"]["calm"]
            t["_vtext"] = vtext_desperate(t["vtext_ratings"])
            trials.append(t)
    return trials


def confusion_quartiles(vint, vtext, y):
    """Cut into 4x4 quartile bins on (vint, vtext); per-cell hack rate."""
    qi = np.digitize(vint, np.quantile(vint, [0.25, 0.5, 0.75])).clip(0, 3)
    qt = np.digitize(vtext, np.quantile(vtext, [0.25, 0.5, 0.75])).clip(0, 3)
    rate = np.full((4, 4), np.nan)
    n = np.zeros((4, 4), dtype=int)
    for i in range(4):
        for j in range(4):
            m = (qi == i) & (qt == j)
            n[i, j] = m.sum()
            if m.sum() > 0:
                rate[i, j] = y[m].mean()
    return rate, n


def conditional_auc(score_a, score_b, y, name_a, name_b, lo_q=0.5):
    """AUC of score_a restricted to bottom-half of score_b (where 'b would have said no')."""
    thr = np.quantile(score_b, lo_q)
    mask = score_b <= thr
    if y[mask].sum() < 3 or y[mask].sum() == mask.sum():
        return None
    auc = roc_auc_score(y[mask], score_a[mask])
    return {"n": int(mask.sum()), "events": int(y[mask].sum()),
            "auc": float(auc),
            "label": f"{name_a} AUC on trials where {name_b} ≤ median"}


def analyze_one(name: str, trials: list[dict]):
    if not trials:
        print(f"\n=== {name}: EMPTY ===")
        return None
    y = np.array([t["_y"] for t in trials])
    vint = np.array([t["_vint"] for t in trials])
    vint_neg_calm = np.array([t["_vint_neg_calm"] for t in trials])
    vtext = np.array([t["_vtext"] for t in trials])
    print(f"\n=== {name} ===")
    print(f"  n={len(y)}  events={int(y.sum())} ({100*y.mean():.1f}%)")

    # AUCs (ground truth check)
    if y.sum() > 2 and y.sum() < len(y):
        a_int = roc_auc_score(y, vint)
        a_neg_calm = roc_auc_score(y, vint_neg_calm)
        a_txt = roc_auc_score(y, vtext)
        print(f"  AUC V_internal[desperate]   = {a_int:.3f}")
        print(f"  AUC V_internal[-calm]       = {a_neg_calm:.3f}")
        print(f"  AUC V_text[desperate-proxy] = {a_txt:.3f}")

    # Correlation between predictors
    r_pearson, p_pearson = stats.pearsonr(vint, vtext)
    r_spearman, p_spearman = stats.spearmanr(vint, vtext)
    r_pearson_nc, _ = stats.pearsonr(vint_neg_calm, vtext)
    print(f"  Pearson r (V_int[desp], V_text)  = {r_pearson:+.3f}  p={p_pearson:.2e}")
    print(f"  Spearman r (V_int[desp], V_text) = {r_spearman:+.3f}  p={p_spearman:.2e}")
    print(f"  Pearson r (V_int[-calm], V_text) = {r_pearson_nc:+.3f}")

    # Quartile heatmap
    rate, n_grid = confusion_quartiles(vint, vtext, y)
    print(f"\n  Hack rate by (V_internal[desperate] quartile × V_text[proxy] quartile):")
    print(f"  rows = V_int quartile (low→high), cols = V_text quartile (low→high)")
    for i in range(4):
        cells = " ".join(f"{rate[i,j]:.2f}({n_grid[i,j]})".rjust(11) for j in range(4))
        print(f"    V_int q{i+1}: {cells}")

    # Diagonal-corner cells (extreme agreement vs disagreement)
    print(f"\n  Diagonal-corner hack rates (extreme agreement vs disagreement):")
    print(f"    AGREE-LOW   (V_int low,  V_text low):  rate={rate[0,0]:.3f}  n={n_grid[0,0]}")
    print(f"    AGREE-HIGH  (V_int high, V_text high): rate={rate[3,3]:.3f}  n={n_grid[3,3]}")
    print(f"    HIDDEN      (V_int high, V_text low):  rate={rate[3,0]:.3f}  n={n_grid[3,0]}  ← 'hidden state' cell")
    print(f"    PERFORMATIVE(V_int low,  V_text high): rate={rate[0,3]:.3f}  n={n_grid[0,3]}  ← 'performative tone' cell")

    # Conditional AUCs
    cond = {}
    cond["vint_on_low_vtext"] = conditional_auc(vint, vtext, y, "V_internal", "V_text")
    cond["vtext_on_low_vint"] = conditional_auc(vtext, vint, y, "V_text", "V_internal")
    cond["vint_neg_calm_on_low_vtext"] = conditional_auc(vint_neg_calm, vtext, y, "V_internal[-calm]", "V_text")
    print(f"\n  Conditional AUCs (could the predictor catch what the other missed?):")
    for k, c in cond.items():
        if c is not None:
            print(f"    {c['label']}  AUC={c['auc']:.3f}  (n={c['n']}, events={c['events']})")

    return {"name": name, "n": int(len(y)), "events": int(y.sum()),
            "aucs": {"vint_desperate": float(a_int) if y.sum() > 2 else None,
                     "vint_neg_calm": float(a_neg_calm) if y.sum() > 2 else None,
                     "vtext_desperate": float(a_txt) if y.sum() > 2 else None},
            "correlation_pearson_vint_vtext": float(r_pearson),
            "correlation_spearman_vint_vtext": float(r_spearman),
            "quartile_hackrate": rate.tolist(),
            "quartile_n": n_grid.tolist(),
            "conditional": {k: c for k, c in cond.items()}}


def main():
    # Datasets
    ext = load(P4_EXT)
    fg = load(P4_FG) + load(PHASE2)
    div_all = load(DIVERSE, judge_field="claude_classification")
    # Drop UNCLEAR for diverse (already done in load via the SHORTCUT/LEGIT filter)

    r1 = analyze_one("D1: Phase 4 extended_unsteered (the strongest H5 dataset)", ext)
    r2 = analyze_one("D2: Phase 4 finegrained + Phase 2 task_a (fast_sum family)", fg)
    # For diverse, restrict to event-positive variants (where we have shortcuts)
    div_pos = [t for t in div_all if t["task_id"] in
               ("hardcoded_lookup_v1", "fake_verifier_v1",
                "silent_spec_drop_v1", "misleading_impl_v1")]
    r3 = analyze_one("D3: phase2_diverse (4 event-positive variants, Claude-judged)",
                     div_pos)

    summary = {"D1_phase4_extended": r1, "D2_phase4_fg_plus_phase2": r2,
               "D3_diverse_claude": r3}
    (OUT / "vint_vtext_divergence.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT/'vint_vtext_divergence.json'}")

    # ============================================================
    # PLOTS
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, res, trials in zip(axes,
                                [r1, r2, r3],
                                [ext, fg, div_pos]):
        if not res:
            continue
        y = np.array([t["_y"] for t in trials])
        vint = np.array([t["_vint"] for t in trials])
        vtext = np.array([t["_vtext"] for t in trials])
        rate = np.array(res["quartile_hackrate"])
        ng = np.array(res["quartile_n"])
        im = ax.imshow(rate, cmap="Reds", vmin=0, vmax=max(0.5, np.nanmax(rate)),
                       aspect="auto", origin="lower")
        for i in range(4):
            for j in range(4):
                if not np.isnan(rate[i, j]):
                    ax.text(j, i, f"{rate[i,j]:.2f}\nn={ng[i,j]}",
                            ha="center", va="center", fontsize=8,
                            color="white" if rate[i, j] > 0.4 else "black")
        ax.set_xticks(range(4)); ax.set_xticklabels(["Q1\nlow", "Q2", "Q3", "Q4\nhigh"], fontsize=8)
        ax.set_yticks(range(4)); ax.set_yticklabels(["Q1\nlow", "Q2", "Q3", "Q4\nhigh"], fontsize=8)
        ax.set_xlabel("V_text [desperate-proxy] quartile")
        ax.set_ylabel("V_internal [desperate] quartile")
        ax.set_title(f"{res['name'][:50]}\nn={res['n']} events={res['events']}  "
                     f"r={res['correlation_pearson_vint_vtext']:+.2f}",
                     fontsize=9)
    fig.suptitle("Hack rate by V_internal × V_text quartile — bottom-left = both 'no', top-right = both 'yes',\n"
                 "top-left = 'hidden state' (V_internal flags, CoT looks calm), "
                 "bottom-right = 'performative tone' (CoT looks desperate, V_internal calm)",
                 fontsize=10, y=1.04)
    plt.tight_layout()
    plt.savefig(OUT / "10_vint_vtext_quartiles.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Wrote plot {OUT/'10_vint_vtext_quartiles.png'}")


if __name__ == "__main__":
    main()
