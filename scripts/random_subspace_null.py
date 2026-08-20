#!/usr/bin/env python3
"""
Random-direction null for the OBSERVATIONAL sweep. Added 2026-08-20 after review.

The paper insists that a steering effect must beat a matched random direction
(section 4.5) and then never applies the same standard to the 50-direction sweep in
section 4.3. That is the paper's own double standard, and it matters here: the
probe scales with activation magnitude and the survivors sit largely on one shared
component, so the live alternative is that ANY 50 directions would produce a similar
number of survivors because they all carry that shared factor.

This runs the identical pipeline on 50 random unit directions drawn from the same
layer, repeated over several draws, and compares survivor counts against the 50
emotion directions on the same trials.

Constraint worth stating plainly: raw activations were retained only for the
80-trial layer-sweep subset, which carries 7 events rather than 14. Both arms are
therefore run on those 80 trials, so the comparison is like-for-like but has less
power than the section 4.3 headline. It bounds the artifact, it does not settle it.

Analysis-only. Writes results/random_subspace_null.json. Runtime ~4 min.
"""
import json
from pathlib import Path

import numpy as np

SEED = 20260819
B = 2000          # per-draw null size; the question is a count comparison, not a tail
RIDGE = 1.0
ALPHA = 0.05
N_DRAWS = 20      # independent sets of 50 random directions
ROOT = Path(__file__).resolve().parent.parent
P4 = ROOT / "results/phase4/llama70b"
OUT = ROOT / "results/random_subspace_null.json"


def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def fit_ll(X1, yy, ridge=RIDGE, iters=100, tol=1e-10):
    pen = np.ones(X1.shape[1]) * ridge
    pen[0] = 0.0
    b = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = np.clip(1.0 / (1.0 + np.exp(-X1 @ b)), 1e-12, 1 - 1e-12)
        W = p * (1 - p)
        H = (X1 * W[:, None]).T @ X1 + np.diag(pen) + 1e-9 * np.eye(X1.shape[1])
        try:
            step = np.linalg.solve(H, X1.T @ (yy - p) - pen * b)
        except np.linalg.LinAlgError:
            return None, False
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    else:
        return None, False
    p = np.clip(1.0 / (1.0 + np.exp(-X1 @ b)), 1e-12, 1 - 1e-12)
    return float((yy * np.log(p) + (1 - yy) * np.log(1 - p)).sum()), True


rows = [r for r in load(P4 / "extended_unsteered_judged.jsonl")
        if r.get("judge_classification") in ("SHORTCUT", "LEGITIMATE") and r.get("emotion_probes")]
y = np.array([1.0 if r["judge_classification"] == "SHORTCUT" else 0.0 for r in rows])
length = np.array([len(r.get("response", "")) for r in rows], float)
zlen = (length - length.mean()) / length.std()
n = len(y)
acts = np.load(P4 / "extended_unsteered_layer_sweep_raw.npz")["layer_53"]
assert acts.shape[0] == n, "activation rows do not align with judged rows"
emos = sorted(rows[0]["emotion_probes"])
EMO = np.array([[float(r["emotion_probes"][e]) for e in emos] for r in rows])
print(f"n = {n}, events = {int(y.sum())}, directions per set = {len(emos)}")
print("NOTE: 7 events, not the 14 of the section 4.3 headline. Power is limited.")


def z(v):
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v * 0.0


def survivors(P):
    """max-T survivor count for one set of directions, conditional null on length."""
    X0 = np.column_stack([np.ones(n), zlen])
    cols = [z(P[:, j]) for j in range(P.shape[1])]
    X1s = [np.column_stack([X0, c]) for c in cols]

    def stats(yy):
        l0, ok = fit_ll(X0, yy)
        if not ok:
            return None
        out = np.empty(len(cols))
        for j, X1 in enumerate(X1s):
            l1, o = fit_ll(X1, yy)
            out[j] = 2 * (l1 - l0) if o else 0.0
        return out

    obs = stats(y)
    if obs is None:
        return None
    b0 = np.linalg.lstsq(X0, y, rcond=None)[0]
    phat = np.clip(X0 @ b0, 1e-3, 1 - 1e-3)
    rng = np.random.default_rng(SEED)
    ge = np.zeros(len(cols))
    eff = 0
    for _ in range(B):
        ys = (rng.random(n) < phat).astype(float)
        if ys.sum() == 0 or ys.sum() == n:
            continue
        s = stats(ys)
        if s is None:
            continue
        eff += 1
        ge += (s.max() >= obs)
    p = (ge + 1) / (eff + 1)
    return int((p < ALPHA).sum()), float(obs.max())


emo_n, emo_max = survivors(EMO)
print(f"\nemotion directions on these 80 trials: {emo_n}/50 survive, max chi2 = {emo_max:.1f}")

rng = np.random.default_rng(SEED)
counts, maxes = [], []
for k in range(N_DRAWS):
    U = rng.normal(size=(acts.shape[1], len(emos)))
    U /= np.linalg.norm(U, axis=0)
    c, m = survivors(acts @ U)
    counts.append(c); maxes.append(m)
    print(f"  random draw {k+1:2d}/{N_DRAWS}: {c}/50 survive, max chi2 = {m:.1f}")

counts = np.array(counts)
R = {"n": n, "events": int(y.sum()), "B": B, "n_draws": N_DRAWS, "seed": SEED,
     "emotion_survivors": emo_n, "emotion_max_chi2": emo_max,
     "random_survivors": counts.tolist(), "random_max_chi2": maxes,
     "random_mean": float(counts.mean()), "random_max": int(counts.max()),
     "p_emotion_exceeds_random": float((counts >= emo_n).mean()),
     "caveat": "80-trial subset, 7 events; raw activations were kept only for this subset"}
print(f"\nrandom sets: mean {counts.mean():.1f}/50, max {counts.max()}/50")
print(f"fraction of random sets reaching the emotion count: {R['p_emotion_exceeds_random']:.2f}")
OUT.write_text(json.dumps(R, indent=1, sort_keys=True))
print(f"wrote {OUT.relative_to(ROOT)}")
