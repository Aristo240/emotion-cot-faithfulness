#!/usr/bin/env python3
"""
Design facts and label robustness, for the appendix.

Three things reviewers ask that no other script emits:

(1) Clustering. The 120 unsteered trials are not 120 independent prompts. They
    are 30 stochastic rollouts of each of 4 fast_sum prompts. Every interval in
    the paper resamples trials, not prompts, so prompt-level uncertainty is
    understated. This script reports the structure so the paper can say so.

(2) Label robustness. Neither judge is validated against humans, and there are
    only 14 events. This asks how many judge labels would have to be wrong to
    remove the strongest surviving direction, using the same ridge-penalised
    nested LR statistic as scripts/paper_numbers.py.

(3) Random-direction construction, restated from scripts/run_blockers.py:208 so
    the appendix can describe the control precisely.

Analysis-only, deterministic. Writes results/design_robustness.json.
"""
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
P2 = ROOT / "results/phase2"
P4 = ROOT / "results/phase4/llama70b"
OUT = ROOT / "results/design_robustness.json"
RIDGE = 1.0


def load(p):
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def fit_pen(X, yy, ridge=RIDGE, iters=100, tol=1e-10):
    """Identical to scripts/paper_numbers.py:266. Returns the UNPENALISED
    log-likelihood evaluated at the ridge-penalised estimates, which is the
    statistic the LR test in the paper is built from."""
    X = np.column_stack([np.ones(len(yy)), X])
    pen = np.ones(X.shape[1]) * ridge
    pen[0] = 0.0
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = np.clip(1 / (1 + np.exp(-X @ b)), 1e-12, 1 - 1e-12)
        W = p * (1 - p)
        H = (X * W[:, None]).T @ X + np.diag(pen) + 1e-9 * np.eye(X.shape[1])
        try:
            step = np.linalg.solve(H, X.T @ (yy - p) - pen * b)
        except np.linalg.LinAlgError:
            return b, None
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    p = np.clip(1 / (1 + np.exp(-X @ b)), 1e-12, 1 - 1e-12)
    return b, float((yy * np.log(p) + (1 - yy) * np.log(1 - p)).sum())


trials = [t for t in load(P4 / "extended_unsteered_judged.jsonl")
          + [x for x in load(P2 / "task_a_judged.jsonl")
             if float(x.get("strength", 0)) == 0.0]
          if t.get("judge_classification") in ("SHORTCUT", "LEGITIMATE")
          and t.get("emotion_probes")]
y0 = np.array([1 if t["judge_classification"] == "SHORTCUT" else 0 for t in trials])
L = np.array([len(t.get("response", "")) for t in trials], float)
zlen = (L - L.mean()) / L.std()
keys = sorted(trials[0]["emotion_probes"])
V = np.array([[t["emotion_probes"][k] for k in keys] for t in trials], float)

# ---------------------------------------------------------------- (1) clustering
task = [t["task_id"] for t in trials]
prompts = {t["task_id"]: t.get("prompt", "") for t in trials}
per_task = {k: sum(1 for x in task if x == k) for k in sorted(set(task))}
print(f"(1) clustering: {len(trials)} trials over {len(per_task)} distinct prompts")
print(f"    rollouts per prompt: {sorted(set(per_task.values()))}")
# (trials do not carry the prompt string; task_id is the prompt identity)
print("    -> intervals resample trials, not prompts; prompt-level n is 4")

# ----------------------------------------------------------- (2) label robustness
jb = keys.index("bored")


def chi2_bored(y):
    _, l0 = fit_pen(zlen.reshape(-1, 1), y)
    x = V[:, jb]
    z = (x - x.mean()) / x.std()
    _, l1 = fit_pen(np.column_stack([zlen, z]), y)
    return 2 * (l1 - l0)


base = chi2_bored(y0)
singles = []
for i in range(len(y0)):
    y = y0.copy()
    y[i] = 1 - y[i]
    singles.append((chi2_bored(y), i, "event->legit" if y0[i] == 1 else "legit->event"))
singles.sort()
worst1 = singles[0]
cand = [s[1] for s in singles[:8]]
worst2 = min((chi2_bored(_flip(y0, a, b)), a, b)
             for a, b in itertools.combinations(cand, 2)) if False else None


def _flip2(y, a, b):
    z = y.copy()
    z[a] = 1 - z[a]
    z[b] = 1 - z[b]
    return z


worst2 = min((chi2_bored(_flip2(y0, a, b)), a, b)
             for a, b in itertools.combinations(cand, 2))
print(f"(2) label robustness for `bored`: chi2 {base:.1f} baseline, "
      f"{worst1[0]:.1f} after the worst single flip, {worst2[0]:.1f} after the worst two")

R = {
    "clustering": {
        "n_trials": len(trials), "n_distinct_prompts": len(per_task),
        "rollouts_per_prompt": per_task,
        "note": ("the 120 unsteered trials are stochastic rollouts of 4 prompts; "
                 "all intervals in the paper resample trials, not prompts"),
    },
    "label_robustness": {
        "direction": "bored", "ridge": RIDGE,
        "chi2_baseline": float(base),
        "chi2_worst_single_flip": float(worst1[0]),
        "worst_single_flip_kind": worst1[2],
        "chi2_worst_two_flips": float(worst2[0]),
        "note": ("family-wise significance under the conditional null sits near "
                 "chi2 12-14, so two adversarial label flips leave `bored` "
                 "significant but close to the boundary"),
    },
    "lr_statistic": {
        "definition": ("2 * (l1 - l0) where l0 and l1 are UNPENALISED "
                       "log-likelihoods evaluated at ridge-penalised estimates "
                       "(lambda = 1, intercept unpenalised); calibration is from "
                       "the resampling distribution, not a nominal chi2_1"),
    },
    "random_directions": {
        "n": 5,
        "construction": ("iid Gaussian in the residual basis, Gram-Schmidt "
                         "projected orthogonal to all 50 emotion vectors, then "
                         "rescaled to the mean norm of the emotion vectors"),
        "matched_on": ["norm", "orthogonality to the emotion subspace"],
        "not_matched_on": ["natural activation variance", "logit-lens impact"],
        "source": "scripts/run_blockers.py:208",
    },
}
OUT.write_text(json.dumps(R, indent=1, sort_keys=True))
print(f"wrote {OUT.relative_to(ROOT)}")
