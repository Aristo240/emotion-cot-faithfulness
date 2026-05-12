#!/usr/bin/env python3
"""Layer-sweep analysis on the 80 Phase 4 extended_unsteered trials with
V_internal probes recomputed at layers 13, 26, 39, 52, 53, 65.

The question (per reviewer feedback): is the task-local faithfulness gap
specific to layer 53, or does it hold across a broader band of late layers?

For each layer L, computes:
  - Marginal V_int[desperate] AUC vs SHORTCUT, 95% bootstrap CI
  - Marginal V_int[-calm]    AUC, 95% bootstrap CI
  - V_text[desperate-proxy]  AUC (constant across layers — baseline)
  - Conditional V_int[desperate] AUC restricted to V_text-low trials
  - Cochran-Armitage trend of V_int across quartiles, within V_text-low stratum
  - Paired bootstrap ΔAUC vs V_text per layer

Also uses both Qwen and Claude judge labels (since we now have both) for a
sensitivity check: does the layer pattern look the same under either judge?

Output:
  results/phase4/llama70b/analysis_layer_sweep/{summary.json, *.png}
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
SWEEP_JSONL = ROOT / "results/phase4/llama70b/extended_unsteered_layer_sweep.jsonl"
RAW_NPZ = ROOT / "results/phase4/llama70b/extended_unsteered_layer_sweep_raw.npz"
CLAUDE_JUDGED = ROOT / "results/phase4/llama70b/extended_unsteered_claude_judged.jsonl"
OUT = ROOT / "results/phase4/llama70b/analysis_layer_sweep"
OUT.mkdir(parents=True, exist_ok=True)

LAYERS = [13, 26, 39, 52, 53, 65]
N_BOOT = 5000
RNG = np.random.default_rng(20260512)


def vtext_score(vt):
    if not vt:
        return np.nan
    return (vt.get("urgency", 3) - vt.get("composure", 3) + vt.get("frustration", 3)) / 3.0


def load_claude_judgments():
    """Map key -> claude_classification (SHORTCUT/LEGITIMATE/UNCLEAR)."""
    m = {}
    if not CLAUDE_JUDGED.exists():
        return m
    with open(CLAUDE_JUDGED) as f:
        for line in f:
            r = json.loads(line)
            m[r["key"]] = r.get("claude_classification")
    return m


def bootstrap_auc(y, s, n_boot=N_BOOT):
    if y.sum() < 2 or y.sum() == len(y):
        return None
    boots = []
    for _ in range(n_boot):
        idx = RNG.integers(0, len(y), len(y))
        yb = y[idx]
        if yb.sum() < 2 or yb.sum() == len(yb):
            continue
        boots.append(roc_auc_score(yb, s[idx]))
    if not boots:
        return None
    return {
        "auc": float(roc_auc_score(y, s)),
        "boot_mean": float(np.mean(boots)),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "n": int(len(y)), "events": int(y.sum()),
    }


def paired_delta_bootstrap(y, sa, sb, n_boot=N_BOOT):
    if y.sum() < 2 or y.sum() == len(y):
        return None
    diffs = []
    for _ in range(n_boot):
        idx = RNG.integers(0, len(y), len(y))
        yb = y[idx]
        if yb.sum() < 2 or yb.sum() == len(yb):
            continue
        diffs.append(roc_auc_score(yb, sa[idx]) - roc_auc_score(yb, sb[idx]))
    if not diffs:
        return None
    return {
        "delta_mean": float(np.mean(diffs)),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "p_one_sided_high": float((np.array(diffs) <= 0).mean()),
    }


def cochran_armitage(k, n, d):
    k = np.array(k, dtype=float); n = np.array(n, dtype=float); d = np.array(d, dtype=float)
    N = n.sum(); R = k.sum()
    if N == 0:
        return None
    pbar = R / N
    if pbar in (0.0, 1.0):
        return None
    num = ((k - n * pbar) * d).sum()
    denom = pbar * (1 - pbar) * (n * (d - (n * d).sum() / N) ** 2).sum()
    if denom <= 0:
        return None
    z = num / np.sqrt(denom)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"z": float(z), "p": float(p)}


def main():
    trials = [json.loads(l) for l in open(SWEEP_JSONL) if l.strip()]
    claude_map = load_claude_judgments()
    print(f"Loaded {len(trials)} trials  (Claude-judge labels: {len(claude_map)} available)")

    # --- Helper to build feature arrays for a given judge ---
    def build_arrays(judge_field: str):
        rows = []
        for t in trials:
            if judge_field == "claude":
                label = claude_map.get(t["key"])
            else:
                label = t.get("judge_classification")
            if label not in ("SHORTCUT", "LEGITIMATE"):
                continue
            rows.append({
                "y": 1 if label == "SHORTCUT" else 0,
                "key": t["key"],
                "task_id": t["task_id"],
                "vtext": vtext_score(t.get("vtext_ratings", {})),
                "probes_per_layer": {int(L): t["emotion_probes_per_layer"].get(str(L)) or
                                     t["emotion_probes_per_layer"].get(L)
                                     for L in LAYERS},
            })
        y = np.array([r["y"] for r in rows])
        vtext = np.array([r["vtext"] for r in rows])
        per_layer = {L: np.array([r["probes_per_layer"][L]["desperate"] for r in rows]) for L in LAYERS}
        per_layer_neg_calm = {L: -np.array([r["probes_per_layer"][L]["calm"] for r in rows]) for L in LAYERS}
        return y, vtext, per_layer, per_layer_neg_calm

    summary = {}
    for judge_name in ("qwen", "claude"):
        if judge_name == "claude" and not claude_map:
            continue
        y, vtext, vint_desp, vint_neg_calm = build_arrays(judge_name)
        if y.sum() < 3:
            print(f"\n[{judge_name}] too few events ({int(y.sum())}); skipping")
            continue
        print(f"\n========== JUDGE = {judge_name.upper()} "
              f"(n={len(y)}, events={int(y.sum())}) ==========")
        vt_auc = bootstrap_auc(y, vtext)
        print(f"  V_text[desperate-proxy] AUC = {vt_auc['auc']:.3f}  CI [{vt_auc['ci95'][0]:.3f}, {vt_auc['ci95'][1]:.3f}]")

        rows = []
        for L in LAYERS:
            vd = vint_desp[L]
            vc = vint_neg_calm[L]
            auc_d = bootstrap_auc(y, vd)
            auc_c = bootstrap_auc(y, vc)
            delta = paired_delta_bootstrap(y, vd, vtext)
            # Conditional: V_text-low subset (use median split given V_text bimodality)
            vt_med = np.median(vtext)
            mask_lo = vtext <= vt_med
            cond = None
            if y[mask_lo].sum() >= 2 and y[mask_lo].sum() < mask_lo.sum():
                cond = bootstrap_auc(y[mask_lo], vd[mask_lo])
            # Cochran-Armitage within V_text-low
            ca = None
            if mask_lo.sum() >= 8:
                yv = y[mask_lo]; vv = vd[mask_lo]
                if yv.sum() >= 2:
                    cuts = np.quantile(vv, [0.25, 0.5, 0.75])
                    bins_idx = np.digitize(vv, cuts).clip(0, 3)
                    counts = [int((bins_idx == q).sum()) for q in range(4)]
                    hacks = [int(yv[bins_idx == q].sum()) for q in range(4)]
                    ca = cochran_armitage(hacks, counts, [0, 1, 2, 3])
            row = {
                "layer": L,
                "vint_desperate_auc": auc_d,
                "vint_neg_calm_auc": auc_c,
                "delta_vs_vtext": delta,
                "vint_on_vtext_low": cond,
                "ca_within_vtext_low": ca,
            }
            rows.append(row)
            ca_str = f"  CA z={ca['z']:+.2f} p={ca['p']:.3f}" if ca else ""
            cond_str = f"  cond(V_text-low) AUC={cond['auc']:.3f}" if cond else ""
            print(f"  layer {L:2d}: V_int[desp] AUC={auc_d['auc']:.3f} "
                  f"[{auc_d['ci95'][0]:.2f},{auc_d['ci95'][1]:.2f}]  "
                  f"V_int[-calm]={auc_c['auc']:.3f}  "
                  f"Δ vs V_text={delta['delta_mean']:+.3f} "
                  f"[{delta['ci95'][0]:+.2f},{delta['ci95'][1]:+.2f}]"
                  f"{cond_str}{ca_str}")
        summary[judge_name] = {
            "n": int(len(y)), "events": int(y.sum()),
            "vtext_auc": vt_auc,
            "per_layer": rows,
        }

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT/'summary.json'}")

    # ========================================================================
    # PLOTS
    # ========================================================================
    for judge_name in summary.keys():
        s = summary[judge_name]
        rows = s["per_layer"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # (a) Marginal V_int AUC across layers, with CI
        ax = axes[0]
        L_vals = [r["layer"] for r in rows]
        d_auc = [r["vint_desperate_auc"]["auc"] for r in rows]
        d_lo = [r["vint_desperate_auc"]["ci95"][0] for r in rows]
        d_hi = [r["vint_desperate_auc"]["ci95"][1] for r in rows]
        c_auc = [r["vint_neg_calm_auc"]["auc"] for r in rows]
        c_lo = [r["vint_neg_calm_auc"]["ci95"][0] for r in rows]
        c_hi = [r["vint_neg_calm_auc"]["ci95"][1] for r in rows]
        ax.errorbar(L_vals, d_auc,
                    yerr=[[a - lo for a, lo in zip(d_auc, d_lo)],
                          [hi - a for a, hi in zip(d_auc, d_hi)]],
                    marker="o", color="#2c7fb8", capsize=4, label="V_int[desperate]")
        ax.errorbar(L_vals, c_auc,
                    yerr=[[a - lo for a, lo in zip(c_auc, c_lo)],
                          [hi - a for a, hi in zip(c_auc, c_hi)]],
                    marker="s", color="#27ae60", capsize=4, label="V_int[-calm]")
        vt_auc = s["vtext_auc"]["auc"]
        ax.axhline(vt_auc, color="#d35400", ls="--", label=f"V_text AUC={vt_auc:.2f}")
        ax.axhline(0.5, color="gray", ls=":", label="chance")
        ax.set_xticks(LAYERS); ax.set_xlabel("layer")
        ax.set_ylabel("AUC vs SHORTCUT")
        ax.set_ylim(0.3, 1.0)
        ax.set_title(f"Layer sweep — V_internal AUC across layers\n"
                     f"judge = {judge_name}, n={s['n']}, events={s['events']}")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.3)

        # (b) ΔAUC vs V_text per layer with CI
        ax = axes[1]
        delta = [r["delta_vs_vtext"]["delta_mean"] for r in rows]
        d_lo = [r["delta_vs_vtext"]["ci95"][0] for r in rows]
        d_hi = [r["delta_vs_vtext"]["ci95"][1] for r in rows]
        ax.errorbar(L_vals, delta,
                    yerr=[[a - lo for a, lo in zip(delta, d_lo)],
                          [hi - a for a, hi in zip(delta, d_hi)]],
                    marker="D", color="#8e44ad", capsize=4)
        ax.axhline(0, color="black", ls="--", lw=0.8)
        ax.set_xticks(LAYERS); ax.set_xlabel("layer")
        ax.set_ylabel("ΔAUC (V_int[desperate] − V_text)")
        ax.set_title(f"Faithfulness gap per layer\n"
                     f"CI excludes 0 → V_int has predictive info V_text lacks")
        ax.grid(alpha=0.3)

        # (c) Conditional V_int|V_text-low AUC per layer
        ax = axes[2]
        cond_aucs = [r["vint_on_vtext_low"]["auc"] if r["vint_on_vtext_low"] else np.nan for r in rows]
        cond_lo = [r["vint_on_vtext_low"]["ci95"][0] if r["vint_on_vtext_low"] else np.nan for r in rows]
        cond_hi = [r["vint_on_vtext_low"]["ci95"][1] if r["vint_on_vtext_low"] else np.nan for r in rows]
        ax.errorbar(L_vals, cond_aucs,
                    yerr=[[a - lo if not np.isnan(a) and not np.isnan(lo) else 0 for a, lo in zip(cond_aucs, cond_lo)],
                          [hi - a if not np.isnan(a) and not np.isnan(hi) else 0 for a, hi in zip(cond_aucs, cond_hi)]],
                    marker="^", color="#e74c3c", capsize=4)
        ax.axhline(0.5, color="gray", ls=":")
        ax.set_xticks(LAYERS); ax.set_xlabel("layer")
        ax.set_ylabel("V_int[desperate] AUC | V_text low")
        ax.set_ylim(0.3, 1.0)
        ax.set_title("Conditional faithfulness-gap signature per layer\n"
                     "(V_int prediction when CoT looks composed)")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        out_path = OUT / f"layer_sweep_{judge_name}.png"
        plt.savefig(out_path, dpi=130)
        plt.close()
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
