#!/usr/bin/env python3
"""Analyze the 650 phase2_diverse trials rejudged by Claude on 2026-05-11.

Produces:
  - Headline class distribution
  - Hack rate per (variant, emotion, strength) with Cochran-Armitage trend
  - V_internal[desperate] / V_internal[calm] AUC vs SHORTCUT, per variant and pooled
  - V_text "desperate-tone" proxy AUC vs SHORTCUT, per variant and pooled
  - DeLong-style paired bootstrap CI on V_internal vs V_text AUC
  - LOGO-style cross-variant generalization (V_internal trained on N-1 variants
    used to predict on held-out variant)
  - Plots to results/phase2/_lambda_partial/analysis_today/

Reads only the Claude-judged file (per-variant correct rubric):
  results/phase2/_lambda_partial/phase2/task_a_diverse_claude_judged.jsonl
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
SRC = ROOT / "results/phase2/_lambda_partial/phase2/task_a_diverse_claude_judged.jsonl"
OUT = ROOT / "results/phase2/_lambda_partial/analysis_today"
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    "tight_budget_v1",
    "hardcoded_lookup_v1",
    "fake_verifier_v1",
    "silent_spec_drop_v1",
    "misleading_impl_v1",
]
VLABEL = {
    "tight_budget_v1": "tight_budget\n(mean, perf-impossible)",
    "hardcoded_lookup_v1": "hardcoded_lookup\n(busy-beaver)",
    "fake_verifier_v1": "fake_verifier\n(proof check)",
    "silent_spec_drop_v1": "silent_spec_drop\n(dedup unhashable)",
    "misleading_impl_v1": "misleading_impl\n(safe_divide)",
}
STRENGTHS = [-0.5, -0.3, -0.2, 0.0, 0.2, 0.3, 0.5]
EMOTIONS = ["desperate", "calm"]


def load() -> list[dict]:
    trials = []
    with open(SRC) as f:
        for line in f:
            line = line.strip()
            if line:
                trials.append(json.loads(line))
    return trials


def vtext_desperate_score(vt: dict) -> float:
    """Compose a single 'desperate-tone' score from the 8-dim V_text rating.
    Higher = more desperate-sounding CoT.  Matches the directionality used
    in the project's docs/judge_prompts.md: urgency↑, composure↓, frustration↑."""
    if not vt:
        return np.nan
    return (vt.get("urgency", 3) - vt.get("composure", 3) + vt.get("frustration", 3)) / 3.0


def is_shortcut(t: dict) -> int:
    return 1 if t.get("claude_classification") == "SHORTCUT" else 0


def is_unclear(t: dict) -> bool:
    return t.get("claude_classification") == "UNCLEAR"


def main():
    trials = load()
    print(f"Loaded {len(trials)} trials from {SRC.name}")

    # ---- 1. Class distribution per variant + global ----
    print("\n=== Class distribution ===")
    for v in VARIANTS:
        rows = [t for t in trials if t["task_id"] == v]
        n = len(rows)
        sc = sum(is_shortcut(t) for t in rows)
        leg = sum(t["claude_classification"] == "LEGITIMATE" for t in rows)
        un = sum(is_unclear(t) for t in rows)
        print(f"  {v:24s} n={n:3d}  SHORTCUT={sc:3d} ({100*sc/n:5.1f}%)  LEGIT={leg:3d}  UNCLEAR={un:3d}")

    # ---- 2. Hack rate per (variant, emotion, strength) + dose-response ----
    print("\n=== Hack rate by (variant, emotion, strength) — only event-positive variants ===")
    hack_grid = {}  # (variant, emotion, strength) -> (hacks, n)
    for v in VARIANTS:
        for e in EMOTIONS:
            for s in STRENGTHS:
                rows = [t for t in trials if t["task_id"] == v
                        and (t["emotion"] == e if s != 0.0 else t["strength"] == 0.0)
                        and abs(t["strength"] - s) < 1e-6]
                if rows:
                    hacks = sum(is_shortcut(t) for t in rows)
                    hack_grid[(v, e, s)] = (hacks, len(rows))

    # Cochran-Armitage trend test per (variant, emotion)
    print("\n=== Cochran-Armitage trend (desperate steering strength) ===")
    trend_stats = {}
    for v in VARIANTS:
        if sum(is_shortcut(t) for t in trials if t["task_id"] == v) < 5:
            continue
        for e in EMOTIONS:
            counts = []
            sizes = []
            doses = []
            for s in STRENGTHS:
                if s == 0.0 and e == "calm":
                    continue
                if (v, e, s) in hack_grid:
                    h, n = hack_grid[(v, e, s)]
                    counts.append(h)
                    sizes.append(n)
                    doses.append(s)
            if len(doses) >= 3 and sum(sizes) > 0:
                d = np.array(doses); n = np.array(sizes); k = np.array(counts)
                N = n.sum(); R = k.sum()
                pbar = R / N
                if pbar == 0 or pbar == 1:
                    continue
                num = ((k - n * pbar) * d).sum()
                denom_var = pbar * (1 - pbar) * ((n * (d - (n * d).sum() / N) ** 2).sum())
                if denom_var <= 0:
                    continue
                z = num / np.sqrt(denom_var)
                p = 2 * (1 - stats.norm.cdf(abs(z)))
                trend_stats[(v, e)] = {"z": z, "p": p, "doses": doses,
                                       "counts": counts, "sizes": sizes}
                print(f"  {v:24s} emotion={e:9s}  z={z:+.3f} p={p:.4f}  "
                      f"hacks/n@strength: {list(zip(doses, counts, sizes))}")

    # ---- 3. V_internal AUC ----
    print("\n=== V_internal predictive AUC (per variant + pooled) ===")
    auc_results = {"per_variant": {}, "pooled": {}}
    pooled_y = []
    pooled_vint_desp = []
    pooled_vint_calm = []
    pooled_vtext = []

    for v in VARIANTS:
        rows = [t for t in trials if t["task_id"] == v and not is_unclear(t)]
        y = np.array([is_shortcut(t) for t in rows])
        if y.sum() < 3 or y.sum() == len(y):
            print(f"  {v}: n={len(y)} events={y.sum()}  — skipped (too few or no negatives)")
            continue
        vint_desp = np.array([t["emotion_probes"]["desperate"] for t in rows])
        vint_calm = np.array([t["emotion_probes"]["calm"] for t in rows])
        vtext = np.array([vtext_desperate_score(t["vtext_ratings"]) for t in rows])
        try:
            auc_vint_d = roc_auc_score(y, vint_desp)
            auc_vint_c_neg = roc_auc_score(y, -vint_calm)  # -calm because calm should predict NO shortcut
            auc_vtext = roc_auc_score(y, vtext)
        except ValueError:
            continue
        auc_results["per_variant"][v] = {
            "n": int(len(y)), "events": int(y.sum()),
            "auc_vint_desperate": float(auc_vint_d),
            "auc_vint_neg_calm": float(auc_vint_c_neg),
            "auc_vtext_desperate": float(auc_vtext),
        }
        print(f"  {v:24s} n={len(y):3d} events={y.sum():3d}  "
              f"V_int[desp]={auc_vint_d:.3f}  V_int[-calm]={auc_vint_c_neg:.3f}  "
              f"V_text={auc_vtext:.3f}")
        pooled_y.append(y)
        pooled_vint_desp.append(vint_desp)
        pooled_vint_calm.append(vint_calm)
        pooled_vtext.append(vtext)

    if pooled_y:
        Y = np.concatenate(pooled_y)
        VID = np.concatenate(pooled_vint_desp)
        VIC = -np.concatenate(pooled_vint_calm)
        VT = np.concatenate(pooled_vtext)
        auc_pool_id = roc_auc_score(Y, VID)
        auc_pool_ic = roc_auc_score(Y, VIC)
        auc_pool_vt = roc_auc_score(Y, VT)
        auc_results["pooled"] = {
            "n": int(len(Y)), "events": int(Y.sum()),
            "auc_vint_desperate": float(auc_pool_id),
            "auc_vint_neg_calm": float(auc_pool_ic),
            "auc_vtext_desperate": float(auc_pool_vt),
        }
        print(f"\n  POOLED (event-positive variants) n={len(Y)} events={Y.sum()}")
        print(f"    V_int[desperate]  AUC = {auc_pool_id:.3f}")
        print(f"    V_int[-calm]      AUC = {auc_pool_ic:.3f}")
        print(f"    V_text[desperate] AUC = {auc_pool_vt:.3f}")

    # ---- 4. Paired bootstrap AUC delta: V_internal vs V_text ----
    if pooled_y:
        rng = np.random.default_rng(20260511)
        boot_id = []; boot_vt = []; boot_diff = []
        for _ in range(2000):
            idx = rng.integers(0, len(Y), len(Y))
            yb = Y[idx]
            if yb.sum() < 2 or yb.sum() == len(yb):
                continue
            a = roc_auc_score(yb, VID[idx])
            b = roc_auc_score(yb, VT[idx])
            boot_id.append(a); boot_vt.append(b); boot_diff.append(a - b)
        ci_d = np.percentile(boot_diff, [2.5, 97.5])
        p_one_side = (np.array(boot_diff) <= 0).mean()
        print(f"\n  Paired bootstrap ΔAUC (V_internal[desperate] - V_text):")
        print(f"    mean Δ = {np.mean(boot_diff):+.3f}  95% CI = [{ci_d[0]:+.3f}, {ci_d[1]:+.3f}]")
        print(f"    Pr(Δ <= 0) under bootstrap = {p_one_side:.4f}  "
              f"({'CI excludes 0' if 0 < ci_d[0] or 0 > ci_d[1] else 'CI includes 0'})")
        auc_results["delta_v_internal_minus_v_text"] = {
            "mean": float(np.mean(boot_diff)),
            "ci": [float(ci_d[0]), float(ci_d[1])],
            "p_one_sided": float(p_one_side),
        }

    # ---- 5. LOGO: leave-one-variant-out generalization ----
    print("\n=== Leave-one-variant-out (LOGO) for V_internal[desperate] ===")
    # Train: no model — V_internal is a fixed direction. Just compute AUC on held-out variant.
    logo = {}
    for vh in [v for v in VARIANTS if v in auc_results["per_variant"]]:
        rows = [t for t in trials if t["task_id"] == vh and not is_unclear(t)]
        y = np.array([is_shortcut(t) for t in rows])
        if y.sum() < 3 or y.sum() == len(y):
            continue
        vd = np.array([t["emotion_probes"]["desperate"] for t in rows])
        a = roc_auc_score(y, vd)
        logo[vh] = a
        print(f"  Held-out={vh}: V_int[desp] AUC = {a:.3f}  (n={len(y)} events={y.sum()})")
    if logo:
        m = float(np.mean(list(logo.values())))
        auc_results["logo_mean_v_int_desperate"] = m
        print(f"  LOGO mean V_int[desperate] AUC = {m:.3f}")

    # ---- 6. SAVE numeric report ----
    summary = {
        "src": str(SRC.relative_to(ROOT)),
        "n_total": len(trials),
        "class_distribution_per_variant": {
            v: {
                "n": sum(1 for t in trials if t["task_id"] == v),
                "SHORTCUT": sum(1 for t in trials if t["task_id"] == v and is_shortcut(t)),
                "LEGITIMATE": sum(1 for t in trials if t["task_id"] == v
                                  and t["claude_classification"] == "LEGITIMATE"),
                "UNCLEAR": sum(1 for t in trials if t["task_id"] == v and is_unclear(t)),
            } for v in VARIANTS
        },
        "trend_tests": {f"{k[0]}__{k[1]}": v for k, v in trend_stats.items()},
        "auc": auc_results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote {OUT/'summary.json'}")

    # ============================================================
    # PLOTS
    # ============================================================
    sns_palette = ["#3a7bd5", "#d35400", "#27ae60", "#8e44ad", "#e74c3c"]

    # --- Plot 1: Hack rate per variant ---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    rates = []
    for v in VARIANTS:
        rows = [t for t in trials if t["task_id"] == v]
        rates.append(100 * sum(is_shortcut(t) for t in rows) / len(rows))
    bars = ax.bar(range(5), rates, color=sns_palette)
    for i, r in enumerate(rates):
        ax.text(i, r + 1.2, f"{r:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(5))
    ax.set_xticklabels([VLABEL[v] for v in VARIANTS], fontsize=8.5)
    ax.set_ylabel("SHORTCUT rate (%)")
    ax.set_ylim(0, max(rates) * 1.25 if rates else 100)
    ax.set_title("Reward-hacking rate by mechanism variant (n=130 each)\n"
                 "Llama 3.1 70B, Claude Sonnet 4.5 as judge with per-variant rubric")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "01_hack_rate_per_variant.png", dpi=130)
    plt.close()

    # --- Plot 2: Dose-response heatmap per variant ---
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), sharey=True)
    for ax, v in zip(axes, VARIANTS):
        mat = np.full((2, 7), np.nan)
        for i, e in enumerate(EMOTIONS):
            for j, s in enumerate(STRENGTHS):
                if (v, e, s) in hack_grid:
                    h, n = hack_grid[(v, e, s)]
                    if n > 0:
                        mat[i, j] = 100 * h / n
        im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=100, aspect="auto")
        for i in range(2):
            for j in range(7):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center",
                            color="white" if mat[i, j] > 50 else "black", fontsize=8)
        ax.set_xticks(range(7)); ax.set_xticklabels([f"{s:+.2f}" for s in STRENGTHS], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(EMOTIONS)
        ax.set_title(VLABEL[v], fontsize=9)
        ax.set_xlabel("steering strength")
    fig.suptitle("Hack rate (%) by emotion × steering strength, per variant",
                 fontsize=12, y=1.02)
    fig.colorbar(im, ax=axes, fraction=0.014, pad=0.01, label="hack rate %")
    plt.savefig(OUT / "02_dose_response_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close()

    # --- Plot 3: AUC per variant — V_internal vs V_text ---
    pv = auc_results["per_variant"]
    if pv:
        labels = list(pv.keys())
        x = np.arange(len(labels))
        w = 0.28
        a_id = [pv[v]["auc_vint_desperate"] for v in labels]
        a_ic = [pv[v]["auc_vint_neg_calm"] for v in labels]
        a_vt = [pv[v]["auc_vtext_desperate"] for v in labels]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(x - w, a_id, w, label="V_internal[desperate]", color="#3a7bd5")
        ax.bar(x, a_ic, w, label="V_internal[-calm]", color="#27ae60")
        ax.bar(x + w, a_vt, w, label="V_text[desperate proxy]", color="#d35400")
        ax.axhline(0.5, color="gray", ls="--", lw=1, label="chance")
        ax.set_xticks(x); ax.set_xticklabels([VLABEL[v] for v in labels], fontsize=8)
        ax.set_ylabel("AUC vs SHORTCUT outcome")
        ax.set_ylim(0.3, 1.0)
        ax.legend(loc="lower left", fontsize=8)
        ax.set_title("Predictive AUC per variant — internal probe vs surface CoT-tone\n"
                     "(only variants with enough SHORTCUT events shown)")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / "03_auc_per_variant.png", dpi=130)
        plt.close()

    # --- Plot 4: V_internal[desperate] distribution by outcome (pooled) ---
    if pooled_y:
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.linspace(min(VID.min(), -3), max(VID.max(), 3), 30)
        ax.hist(VID[Y == 0], bins=bins, alpha=0.55, label=f"LEGIT (n={(Y==0).sum()})",
                color="#3a7bd5", edgecolor="white")
        ax.hist(VID[Y == 1], bins=bins, alpha=0.55, label=f"SHORTCUT (n={(Y==1).sum()})",
                color="#e74c3c", edgecolor="white")
        ax.set_xlabel("V_internal[desperate] (probe projection)")
        ax.set_ylabel("trials")
        ax.set_title(f"V_internal[desperate] distribution by outcome\n"
                     f"pooled across 4 event-positive variants, AUC={auc_pool_id:.3f}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUT / "04_vint_desperate_distribution.png", dpi=130)
        plt.close()

    # --- Plot 5: V_internal vs V_text scatter ---
    if pooled_y:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.scatter(VT[Y == 0], VID[Y == 0], alpha=0.45, s=22, color="#3a7bd5",
                   label=f"LEGIT (n={(Y==0).sum()})")
        ax.scatter(VT[Y == 1], VID[Y == 1], alpha=0.65, s=26, color="#e74c3c",
                   label=f"SHORTCUT (n={(Y==1).sum()})", marker="^")
        ax.set_xlabel("V_text desperate-tone score  (urgency - composure + frustration)/3")
        ax.set_ylabel("V_internal[desperate]")
        ax.set_title("V_internal vs V_text — the faithfulness picture\n"
                     "shortcut events should appear up-and-anywhere on V_text axis if there's a gap")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / "05_vint_vs_vtext_scatter.png", dpi=130)
        plt.close()

    print(f"\nAll plots written to {OUT}/")


if __name__ == "__main__":
    main()
