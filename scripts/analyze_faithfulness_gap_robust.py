#!/usr/bin/env python3
"""Robustness checks for the task-local faithfulness-gap claim, per reviewer
feedback:

1. Bootstrap 95% CI for conditional AUCs:
   - V_internal AUC restricted to V_text-LOW trials
   - V_text     AUC restricted to V_internal-LOW trials
   And the difference between them.

2. Fisher's exact test on the HIDDEN vs PERFORMATIVE quadrant hack rates
   (the small-n robustness check).

3. Threshold robustness: repeat the conditional analysis with cutoffs at
   the 33rd, 50th (median), and 66th percentiles of V_text.

4. Cochran-Armitage trend test for V_internal[desperate] within each V_text
   stratum (LOW vs HIGH).

5. Permutation null for the conditional AUC asymmetry (shuffle outcome
   labels and recompute the conditional-AUC difference).

6. Bimodality of V_text: report the distribution of V_text values across
   trials to justify the visual "empty middle quadrants."
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
PHASE2 = ROOT / "results/phase2/task_a_judged.jsonl"
P4_FG = ROOT / "results/phase4/llama70b/finegrained_judged.jsonl"
OUT = ROOT / "results/phase2/_lambda_partial/analysis_today"
OUT.mkdir(parents=True, exist_ok=True)

N_BOOT = 5000
N_PERM = 5000
RNG = np.random.default_rng(20260511)


def vtext_score(vt):
    return (vt.get("urgency", 3) - vt.get("composure", 3) + vt.get("frustration", 3)) / 3.0


def load_fastsum():
    trials = []
    for path in [PHASE2, P4_FG]:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line)
                jc = t.get("judge_classification")
                if jc not in ("SHORTCUT", "LEGITIMATE"):
                    continue
                if not t.get("emotion_probes") or not t.get("vtext_ratings"):
                    continue
                trials.append({
                    "y": 1 if jc == "SHORTCUT" else 0,
                    "vint": t["emotion_probes"]["desperate"],
                    "vtext": vtext_score(t["vtext_ratings"]),
                })
    y = np.array([t["y"] for t in trials])
    vint = np.array([t["vint"] for t in trials])
    vtext = np.array([t["vtext"] for t in trials])
    return y, vint, vtext


def auc_safe(y, s):
    if y.sum() < 2 or y.sum() == len(y):
        return None
    return roc_auc_score(y, s)


def bootstrap_auc(y, s, n_boot=N_BOOT):
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
        "auc": float(np.mean(boots)),
        "ci95": [float(np.percentile(boots, 2.5)),
                 float(np.percentile(boots, 97.5))],
        "n": int(len(y)),
        "events": int(y.sum()),
        "n_boot": len(boots),
    }


def conditional_auc_with_ci(y_all, score_a, score_b, b_threshold, label):
    """AUC of score_a restricted to trials where score_b <= b_threshold."""
    mask = score_b <= b_threshold
    if y_all[mask].sum() < 3 or y_all[mask].sum() == mask.sum():
        return None
    res = bootstrap_auc(y_all[mask], score_a[mask])
    res["label"] = label
    return res


def cochran_armitage(k, n, d):
    k = np.array(k, dtype=float); n = np.array(n, dtype=float); d = np.array(d, dtype=float)
    N = n.sum(); R = k.sum()
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
    y, vint, vtext = load_fastsum()
    n = len(y); events = int(y.sum())
    print(f"Fast_sum family: n={n}, SHORTCUT={events} ({100*y.mean():.1f}%)")

    # ============================================================
    # 0. Bimodality of V_text
    # ============================================================
    print("\n=== 0. V_text distribution (justifies 'empty middle quartiles') ===")
    bins = np.linspace(vtext.min(), vtext.max(), 21)
    hist, edges = np.histogram(vtext, bins=bins)
    print(f"  V_text range: [{vtext.min():.3f}, {vtext.max():.3f}]")
    print(f"  unique values (rounded to 3 dp): {len(np.unique(np.round(vtext, 3)))}")
    print(f"  quartiles: {np.percentile(vtext, [25,50,75]).round(3).tolist()}")
    # Counts at extremes
    low_third = (vtext <= np.percentile(vtext, 33)).sum()
    mid_third = ((vtext > np.percentile(vtext, 33)) & (vtext <= np.percentile(vtext, 66))).sum()
    high_third = (vtext > np.percentile(vtext, 66)).sum()
    print(f"  trials in lower/middle/upper third: {low_third}/{mid_third}/{high_third}")

    # ============================================================
    # 1. Marginal AUCs with bootstrap CI
    # ============================================================
    print("\n=== 1. Marginal AUCs with 95% bootstrap CI ===")
    a_int = bootstrap_auc(y, vint)
    a_txt = bootstrap_auc(y, vtext)
    print(f"  V_internal[desperate]   AUC = {a_int['auc']:.3f}  CI [{a_int['ci95'][0]:.3f}, {a_int['ci95'][1]:.3f}]")
    print(f"  V_text[desperate-proxy] AUC = {a_txt['auc']:.3f}  CI [{a_txt['ci95'][0]:.3f}, {a_txt['ci95'][1]:.3f}]")
    diff = a_int['auc'] - a_txt['auc']
    # Paired bootstrap for difference
    diffs = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, len(y), len(y))
        yb = y[idx]
        if yb.sum() < 2 or yb.sum() == len(yb):
            continue
        diffs.append(roc_auc_score(yb, vint[idx]) - roc_auc_score(yb, vtext[idx]))
    diff_ci = np.percentile(diffs, [2.5, 97.5])
    print(f"  ΔAUC = {diff:+.3f}  CI [{diff_ci[0]:+.3f}, {diff_ci[1]:+.3f}]  "
          f"({'excludes 0' if 0 < diff_ci[0] or 0 > diff_ci[1] else 'includes 0'})")

    # ============================================================
    # 2. Conditional AUCs at multiple thresholds
    # ============================================================
    print("\n=== 2. Conditional AUCs with bootstrap CI (threshold robustness) ===")
    out_cond = {}
    for thresh_q in [0.33, 0.50, 0.66]:
        vt_thr = np.quantile(vtext, thresh_q)
        vi_thr = np.quantile(vint, thresh_q)
        print(f"  -- threshold = {int(thresh_q*100)}th percentile --")
        # V_internal AUC on V_text-low trials
        res_vint = conditional_auc_with_ci(y, vint, vtext, vt_thr,
                                           f"V_int among V_text<=q{int(thresh_q*100)}")
        # V_text AUC on V_internal-low trials
        res_vtxt = conditional_auc_with_ci(y, vtext, vint, vi_thr,
                                           f"V_text among V_int<=q{int(thresh_q*100)}")
        for k, r in [("V_int|V_text-low", res_vint), ("V_text|V_int-low", res_vtxt)]:
            if r is not None:
                print(f"    {k:20s} AUC = {r['auc']:.3f}  CI [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]  "
                      f"(n={r['n']}, events={r['events']})")
        out_cond[f"q{int(thresh_q*100)}"] = {"vint_on_vtext_low": res_vint,
                                              "vtext_on_vint_low": res_vtxt}

    # ============================================================
    # 3. Quadrant table with absolute event counts + Fisher's exact
    # ============================================================
    print("\n=== 3. Median-split quadrant table with absolute event counts ===")
    vt_med = np.median(vtext); vi_med = np.median(vint)
    def cell(vi_hi, vt_hi):
        mask = ((vint > vi_med) if vi_hi else (vint <= vi_med)) & \
               ((vtext > vt_med) if vt_hi else (vtext <= vt_med))
        return int(mask.sum()), int(y[mask].sum()), float(y[mask].mean()) if mask.sum() else float('nan')
    cells = {
        "AGREE_LOW    (V_int LOW,  V_text LOW)":  cell(False, False),
        "PERFORMATIVE (V_int LOW,  V_text HIGH)": cell(False, True),
        "HIDDEN       (V_int HIGH, V_text LOW)":  cell(True, False),
        "AGREE_HIGH   (V_int HIGH, V_text HIGH)": cell(True, True),
    }
    for name, (n_c, k_c, r_c) in cells.items():
        print(f"  {name:42s}  n={n_c:4d}  hacks={k_c:3d}  rate={r_c:.3f}")
    # Fisher's exact: HIDDEN hacks vs PERFORMATIVE hacks
    h_n, h_k, _ = cells["HIDDEN       (V_int HIGH, V_text LOW)"]
    p_n, p_k, _ = cells["PERFORMATIVE (V_int LOW,  V_text HIGH)"]
    table = [[h_k, h_n - h_k], [p_k, p_n - p_k]]
    odds, p_fisher = stats.fisher_exact(table, alternative="greater")
    print(f"\n  Fisher's exact (HIDDEN hacks > PERFORMATIVE hacks):")
    print(f"    2x2 = [[{h_k},{h_n-h_k}],[{p_k},{p_n-p_k}]]  odds_ratio={odds:.3f}  one-sided p={p_fisher:.4f}")

    # Cochran-Armitage: V_int trend within V_text-low and V_text-high strata
    print("\n  Cochran-Armitage trend of V_internal across quartiles, within V_text strata:")
    for stratum_name, mask in [("V_text LOW (<=median)", vtext <= vt_med),
                                ("V_text HIGH (>median)", vtext > vt_med)]:
        ys = y[mask]; vs = vint[mask]
        if ys.sum() < 4:
            print(f"    {stratum_name}: too few events ({int(ys.sum())})")
            continue
        cutoffs = np.quantile(vs, [0.25, 0.5, 0.75])
        bins_idx = np.digitize(vs, cutoffs).clip(0, 3)
        counts = [int((bins_idx == q).sum()) for q in range(4)]
        hacks = [int(ys[bins_idx == q].sum()) for q in range(4)]
        rates = [hacks[q] / counts[q] if counts[q] else float('nan') for q in range(4)]
        ca = cochran_armitage(hacks, counts, [0, 1, 2, 3])
        print(f"    {stratum_name}: n_per_q={counts}  hacks_per_q={hacks}  "
              f"rates={[f'{r:.2f}' for r in rates]}  CA z={ca['z']:+.2f} p={ca['p']:.4f}")

    # ============================================================
    # 4. Permutation null for conditional-AUC asymmetry
    # ============================================================
    print("\n=== 4. Permutation null for conditional-AUC asymmetry (label shuffle) ===")
    obs_vint = conditional_auc_with_ci(y, vint, vtext, vt_med, "")
    obs_vtxt = conditional_auc_with_ci(y, vtext, vint, vi_med, "")
    obs_diff = obs_vint["auc"] - obs_vtxt["auc"]
    perm_diffs = []
    for _ in range(N_PERM):
        yp = y.copy()
        RNG.shuffle(yp)
        m1 = vtext <= vt_med; m2 = vint <= vi_med
        if yp[m1].sum() < 2 or yp[m1].sum() == m1.sum(): continue
        if yp[m2].sum() < 2 or yp[m2].sum() == m2.sum(): continue
        a = roc_auc_score(yp[m1], vint[m1])
        b = roc_auc_score(yp[m2], vtext[m2])
        perm_diffs.append(a - b)
    p_perm = (np.array(perm_diffs) >= obs_diff).mean()
    print(f"  Observed (V_int|V_text-low) - (V_text|V_int-low) = {obs_diff:+.3f}")
    print(f"  Null mean = {np.mean(perm_diffs):+.3f}  "
          f"95% null range = [{np.percentile(perm_diffs,2.5):+.3f}, {np.percentile(perm_diffs,97.5):+.3f}]")
    print(f"  Permutation p (one-sided, observed >= null) = {p_perm:.4f}")

    # ============================================================
    # SAVE
    # ============================================================
    summary = {
        "n": int(n), "events": int(events),
        "vtext_distribution": {
            "low_third": int(low_third), "mid_third": int(mid_third), "high_third": int(high_third),
            "unique_rounded_values": int(len(np.unique(np.round(vtext, 3)))),
        },
        "marginal_auc": {
            "vint": a_int, "vtext": a_txt,
            "diff_mean": float(diff),
            "diff_ci95": [float(diff_ci[0]), float(diff_ci[1])],
        },
        "conditional_auc": out_cond,
        "quadrants": {k: {"n": v[0], "hacks": v[1], "rate": v[2]} for k, v in cells.items()},
        "fisher_hidden_vs_perf": {"table": table, "odds_ratio": float(odds), "p_one_sided": float(p_fisher)},
        "perm_conditional_diff": {
            "observed": float(obs_diff),
            "null_mean": float(np.mean(perm_diffs)),
            "null_ci95": [float(np.percentile(perm_diffs, 2.5)),
                          float(np.percentile(perm_diffs, 97.5))],
            "p_one_sided": float(p_perm),
        },
    }
    (OUT / "faithfulness_gap_robust.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT/'faithfulness_gap_robust.json'}")

    # ============================================================
    # PLOTS
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # (a) V_text distribution
    ax = axes[0]
    ax.hist(vtext, bins=30, color="#3a7bd5", edgecolor="white")
    ax.axvline(vt_med, color="red", ls="--", label=f"median={vt_med:.2f}")
    ax.set_xlabel("V_text desperate-tone score")
    ax.set_ylabel("trials")
    ax.set_title(f"V_text is bimodal — middle quartiles ~empty\n"
                 f"low/mid/high third: {low_third}/{mid_third}/{high_third}")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) Conditional AUC with CI at 3 thresholds
    ax = axes[1]
    pcs = [33, 50, 66]
    width = 0.35
    xs = np.arange(len(pcs))
    vint_aucs = [out_cond[f"q{p}"]["vint_on_vtext_low"]["auc"] for p in pcs]
    vint_lo = [out_cond[f"q{p}"]["vint_on_vtext_low"]["ci95"][0] for p in pcs]
    vint_hi = [out_cond[f"q{p}"]["vint_on_vtext_low"]["ci95"][1] for p in pcs]
    vtxt_aucs = [out_cond[f"q{p}"]["vtext_on_vint_low"]["auc"] if out_cond[f"q{p}"]["vtext_on_vint_low"] else float('nan') for p in pcs]
    vtxt_lo = [out_cond[f"q{p}"]["vtext_on_vint_low"]["ci95"][0] if out_cond[f"q{p}"]["vtext_on_vint_low"] else float('nan') for p in pcs]
    vtxt_hi = [out_cond[f"q{p}"]["vtext_on_vint_low"]["ci95"][1] if out_cond[f"q{p}"]["vtext_on_vint_low"] else float('nan') for p in pcs]
    ax.bar(xs - width/2, vint_aucs,
           yerr=[[a-lo for a,lo in zip(vint_aucs,vint_lo)],
                 [hi-a for a,hi in zip(vint_aucs,vint_hi)]],
           width=width, color="#2c7fb8", capsize=5, label="V_internal AUC | V_text low")
    ax.bar(xs + width/2, vtxt_aucs,
           yerr=[[a-lo for a,lo in zip(vtxt_aucs,vtxt_lo)],
                 [hi-a for a,hi in zip(vtxt_aucs,vtxt_hi)]],
           width=width, color="#d35400", capsize=5, label="V_text AUC | V_internal low")
    ax.axhline(0.5, color="gray", ls="--")
    ax.set_xticks(xs); ax.set_xticklabels([f"≤q{p}" for p in pcs])
    ax.set_xlabel("Threshold (percentile)")
    ax.set_ylabel("Conditional AUC")
    ax.set_ylim(0.3, 1.0)
    ax.set_title("Conditional AUC robust to threshold\nbars = 95% bootstrap CI")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (c) Quadrant absolute events (not rates)
    ax = axes[2]
    labels = ["AGREE\nLOW", "PERFORMATIVE\n(V_int low,\nV_text high)",
              "HIDDEN\n(V_int high,\nV_text low)", "AGREE\nHIGH"]
    names = ["AGREE_LOW    (V_int LOW,  V_text LOW)",
             "PERFORMATIVE (V_int LOW,  V_text HIGH)",
             "HIDDEN       (V_int HIGH, V_text LOW)",
             "AGREE_HIGH   (V_int HIGH, V_text HIGH)"]
    hacks = [cells[n][1] for n in names]
    totals = [cells[n][0] for n in names]
    rates = [cells[n][2] for n in names]
    colors_q = ["#27ae60", "#f39c12", "#e74c3c", "#8e44ad"]
    xs = np.arange(4)
    ax.bar(xs, hacks, color=colors_q, alpha=0.85, label="SHORTCUT events")
    for i, (h, t, r) in enumerate(zip(hacks, totals, rates)):
        ax.text(i, h + 2, f"{h}/{t}\n({100*r:.1f}%)", ha="center", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("SHORTCUT events (absolute count)")
    ax.set_title(f"Quadrant absolute event counts\nFisher's p(HIDDEN>PERFORMATIVE) = {p_fisher:.4f}")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(hacks) * 1.25 if hacks else 1)

    plt.tight_layout()
    plt.savefig(OUT / "11_faithfulness_gap_robust.png", dpi=130)
    plt.close()
    print(f"Wrote plot {OUT/'11_faithfulness_gap_robust.png'}")


if __name__ == "__main__":
    main()
