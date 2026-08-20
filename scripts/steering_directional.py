#!/usr/bin/env python3
"""
Directional steering analysis and manipulation check. Added 2026-08-20 after
review.

Two problems with the pooled arm the paper reported as "Emotion vector @ +-0.3":

  1. It pools BOTH SIGNS. The claim under audit is directional, that increasing
     a "desperate" direction increases reward hacking. If +0.3 raised the rate
     and -0.3 lowered it, the pooled rate could be unchanged while the vector
     was strongly causal. Pooling cannot test the claim.
  2. It pools TWO EMOTIONS, `desperate` and `calm`. That is defensible for a
     specificity contrast against random directions, but it is not a test of the
     registered direction, and the manuscript's singular phrasing implied it was.

This script reports every cell separately, runs the directional contrasts, and
adds the manipulation check that was missing: whether steering actually moved the
intended representation. Without that check a behavioural null is ambiguous
between "the representation does not drive the behaviour" and "the intervention
did not move the representation".

Analysis-only, no GPU, no network. Writes results/steering_directional.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
P2 = ROOT / "results/phase2"
P4 = ROOT / "results/phase4/llama70b"
OUT = ROOT / "results/steering_directional.json"
LABELS = ("SHORTCUT", "LEGITIMATE")


def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


A = [t for t in load(P2 / "task_a_judged.jsonl") if t.get("judge_classification") in LABELS]
R = [t for t in load(P4 / "random_directions_judged.jsonl")
     if t.get("judge_classification") in LABELS]


def cell(rows, emotion=None, strength=None):
    sel = rows
    if emotion is not None:
        sel = [r for r in sel if r.get("emotion") == emotion]
    if strength is not None:
        sel = [r for r in sel if abs(float(r.get("strength", 0)) - strength) < 1e-9]
    return sum(r["judge_classification"] == "SHORTCUT" for r in sel), len(sel)


def rr_ci(a, b):
    """Risk ratio with a Katz log CI. None when a cell is empty."""
    if a[0] == 0 or b[0] == 0:
        return None
    rr = (a[0] / a[1]) / (b[0] / b[1])
    se = np.sqrt(1 / a[0] - 1 / a[1] + 1 / b[0] - 1 / b[1])
    return {"rr": float(rr), "ci95": [float(np.exp(np.log(rr) - 1.96 * se)),
                                      float(np.exp(np.log(rr) + 1.96 * se))]}


def fisher(a, b):
    o, p = fisher_exact([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]])
    return {"odds_ratio": float(o), "p": float(p)}


R_ = {}
baseline = (14, 120)

# ---------------------------------------------- what the pooled arm contains
cells = {}
for emo in ("desperate", "calm"):
    for s in (0.30, -0.30):
        cells[f"{emo}@{s:+.2f}"] = list(cell(A, emo, s))
for s in (0.30, -0.30):
    cells[f"random@{s:+.2f}"] = list(cell(R, None, s))
R_["cells"] = cells
R_["baseline"] = list(baseline)
print("cells behind the pooled arm:")
for k, v in cells.items():
    print(f"  {k:<18} {v[0]:>2}/{v[1]:<4} = {v[0]/v[1]:.3f}")

dp, dm = cell(A, "desperate", 0.30), cell(A, "desperate", -0.30)
rp, rm = cell(R, None, 0.30), cell(R, None, -0.30)

# ---------------------------------------------- the directional contrasts
R_["directional"] = {
    "desperate_plus_vs_random_plus": {**fisher(dp, rp), **(rr_ci(dp, rp) or {})},
    "desperate_plus_vs_baseline": {**fisher(dp, baseline), **(rr_ci(dp, baseline) or {})},
    "desperate_minus_vs_baseline": {**fisher(dm, baseline), **(rr_ci(dm, baseline) or {})},
}
print("\ndirectional contrasts:")
for k, v in R_["directional"].items():
    ci = f", RR {v['rr']:.2f} 95% CI [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]" if "rr" in v else ""
    print(f"  {k}: OR={v['odds_ratio']:.2f}, Fisher p={v['p']:.3f}{ci}")

# ---------------------------------------------- manipulation check
# Did steering move the projection it was supposed to move? The readout window
# still spans the response, so this shows the intervention had a measurable
# effect on the probed quantity, not that it acted only before generation.
AP = [t for t in load(P2 / "task_a_judged.jsonl") if t.get("emotion_probes")]
un = np.array([float(r["emotion_probes"]["desperate"]) for r in AP
               if abs(float(r.get("strength", 0))) < 1e-9])
man = {"unsteered_mean": float(un.mean()), "unsteered_n": int(len(un)), "by_strength": {}}
print(f"\nmanipulation check (desperate projection, unsteered mean {un.mean():+.4f}):")
for s in (-0.50, -0.30, 0.30, 0.50):
    v = np.array([float(r["emotion_probes"]["desperate"]) for r in AP
                  if r.get("emotion") == "desperate"
                  and abs(float(r.get("strength", 0)) - s) < 1e-9])
    if not len(v):
        continue
    p = float(mannwhitneyu(v, un, alternative="two-sided").pvalue)
    man["by_strength"][f"{s:+.2f}"] = {"mean": float(v.mean()), "n": int(len(v)), "p_vs_unsteered": p}
    print(f"  s={s:+.2f}: mean {v.mean():+.4f} (n={len(v)}), p={p:.3g}")
# sort numerically, not as strings, or "+0.30" precedes "-0.50"
means = [man["by_strength"][k]["mean"] for k in sorted(man["by_strength"], key=float)]
man["monotone_in_strength"] = bool(all(x < y for x, y in zip(means, means[1:])))
man["moved_at_plus_0_3"] = bool(man["by_strength"]["+0.30"]["p_vs_unsteered"] < 0.05
                                and man["by_strength"]["+0.30"]["mean"] > un.mean())
R_["manipulation_check"] = man
print(f"  monotone in strength: {man['monotone_in_strength']};"
      f" moved at +0.3: {man['moved_at_plus_0_3']}")

OUT.write_text(json.dumps(R_, indent=1, sort_keys=True))
print(f"\nwrote {OUT.relative_to(ROOT)}")
