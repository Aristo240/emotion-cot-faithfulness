#!/usr/bin/env python3
"""
Does the bootstrap design matter? (appendix check)

The reported AUC intervals use an outcome-stratified bootstrap: positives and
negatives are resampled separately, so the interval is conditional on the observed
14 events. That is standard for an AUC but it conditions on a quantity the design
did not fix.

What the design DID fix is 30 generations for each of 4 prompt variants. The
design-consistent resample therefore draws 30 with replacement within each prompt
and lets the event count vary. With per-prompt rates of 4/30, 0/30, 9/30 and 1/30
that is worth checking rather than assuming, which is what this script does.

Analysis-only, deterministic. Writes results/bootstrap_design.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

SEED = 20260819
B = 5000
ROOT = Path(__file__).resolve().parent.parent
P2 = ROOT / "results/phase2"
P4 = ROOT / "results/phase4/llama70b"
OUT = ROOT / "results/bootstrap_design.json"


def load(p):
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


trials = [t for t in load(P4 / "extended_unsteered_judged.jsonl")
          + [x for x in load(P2 / "task_a_judged.jsonl")
             if float(x.get("strength", 0)) == 0.0]
          if t.get("judge_classification") in ("SHORTCUT", "LEGITIMATE")
          and t.get("emotion_probes")]
y = np.array([1 if t["judge_classification"] == "SHORTCUT" else 0 for t in trials])
task = np.array([t["task_id"] for t in trials])
d = np.array([float(t["emotion_probes"]["desperate"]) for t in trials])
L = np.array([len(t.get("response", "")) for t in trials], float)
GROUPS = [np.where(task == t)[0] for t in sorted(set(task))]


def auc(x, yy):
    if len(np.unique(yy)) < 2:
        return float("nan")
    r = rankdata(x)
    n1 = yy.sum()
    n0 = len(yy) - n1
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def draw(rng, mode):
    if mode == "outcome":
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        return np.concatenate([rng.choice(pos, len(pos), True),
                               rng.choice(neg, len(neg), True)])
    return np.concatenate([rng.choice(g, len(g), True) for g in GROUPS])


def ci(x, mode):
    rng = np.random.default_rng(SEED)
    v = [auc(x[i], y[i]) for i in (draw(rng, mode) for _ in range(B))]
    v = np.array([q for q in v if not np.isnan(q)])
    return {"lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5)),
            "usable_draws": int(len(v)), "B": B}


def ci_delta(mode):
    rng = np.random.default_rng(SEED)
    v = []
    for _ in range(B):
        i = draw(rng, mode)
        a, b = auc(L[i], y[i]), auc(d[i], y[i])
        if not (np.isnan(a) or np.isnan(b)):
            v.append(a - b)
    v = np.array(v)
    return {"mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)),
            "hi": float(np.percentile(v, 97.5)), "usable_draws": int(len(v)), "B": B}


R = {"per_prompt": {t: [int(y[task == t].sum()), int((task == t).sum())]
                    for t in sorted(set(task))},
     "point": {"desperate": auc(d, y), "length": auc(L, y)},
     "desperate": {m: ci(d, m) for m in ("outcome", "prompt")},
     "length": {m: ci(L, m) for m in ("outcome", "prompt")},
     "delta_length_minus_desperate": {m: ci_delta(m) for m in ("outcome", "prompt")},
     "note": ("outcome = positives and negatives resampled separately (reported in "
              "the paper); prompt = 30 drawn with replacement within each of the 4 "
              "prompt strata, which is what the design fixed")}
for k in ("desperate", "length"):
    o, p = R[k]["outcome"], R[k]["prompt"]
    print(f"{k:<12} point {R['point'][k]:.3f}  outcome [{o['lo']:.3f}, {o['hi']:.3f}]"
          f"  prompt [{p['lo']:.3f}, {p['hi']:.3f}]")
dl = R["delta_length_minus_desperate"]
for m in ("outcome", "prompt"):
    print(f"delta ({m:<7}) {dl[m]['mean']:+.3f} [{dl[m]['lo']:+.3f}, {dl[m]['hi']:+.3f}]")
R["max_ci_shift"] = max(abs(R[k][m2][b] - R[k]["outcome"][b])
                        for k in ("desperate", "length")
                        for m2 in ("prompt",) for b in ("lo", "hi"))
print(f"largest endpoint shift between designs: {R['max_ci_shift']:.4f}")
OUT.write_text(json.dumps(R, indent=1, sort_keys=True))
print(f"wrote {OUT.relative_to(ROOT)}")
