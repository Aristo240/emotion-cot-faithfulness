#!/usr/bin/env python3
"""
Validity audit of section 4.1 -- the paper's one SUPPORTED verdict.

Design and justification: scripts/_emobank_validity_design_notes.md

Section 6 of the manuscript tells the field to carry a simple output baseline.
Table 1's one SUPPORTED row carries only TF-IDF. This closes that gap and, while
the fold machinery is open, runs the other validity checks the same data
licenses. Eight blocks, one per validity type. Analysis-only, ~15 min, no GPU.

Appends to results/emobank_validity.json. Does NOT touch
results/emobank_baseline.json -- the 400-assertion gate reads that file and its
existing values must not move.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parent.parent
TRIALS = ROOT / "results/emobank/llama70b/trials.jsonl"
OUT = ROOT / "results/emobank_validity.json"
DIMS = ["V", "A", "D"]
SEED = 20260510
ALPHAS = [0.1, 1, 10, 100]

rows = [json.loads(l) for l in open(TRIALS) if l.strip()]
text = [r["text"] for r in rows]
Y = np.array([[r[d] for d in DIMS] for r in rows], float)
emos = sorted(rows[0]["emotion_probes"])
Xp = np.array([[r["emotion_probes"][e] for e in emos] for r in rows], float)
# EmoBank ids are "<document>_<start>_<end>".
docs = np.array([r["id"].rsplit("_", 2)[0] for r in rows])
n = len(rows)
print(f"n = {n} sentences, {len(emos)} probes, {len(set(docs))} documents")

# ---- feature builders -------------------------------------------------------
norm = np.linalg.norm(Xp, axis=1, keepdims=True)
Xp_unit = Xp / norm   # NOT 'magnitude removed': EmoBank values are already
                      # cosines (run_emobank_validation.py:136 divides by
                      # ||r||). This removes the 50-vector's remaining scale
                      # and is the ONLY readout commensurable with the trials,
                      # where the same operation cancels ||a||. See the
                      # addendum in _emobank_validity_design_notes.md.
n_chars = np.array([len(t) for t in text], float)
n_words = np.array([len(t.split()) for t in text], float)
# 'probe_norm' here is ||P r||/||r||: the norm of the 50 COSINES, not ||r||,
# which was never stored. It is a scale feature of the probe vector, and is
# labelled as such rather than as activation magnitude.
Xtriv = np.column_stack([n_chars, n_words, norm[:, 0]])


def f_probe(tr, te):      return Xp[tr], Xp[te]
def f_probe_unit(tr, te): return Xp_unit[tr], Xp_unit[te]
def f_trivial(tr, te):    return Xtriv[tr], Xtriv[te]


def _tfidf(tr, te, max_features=8000, min_df=1):
    """Fitted inside the fold: the vocabulary never sees held-out sentences."""
    v = TfidfVectorizer(lowercase=True, ngram_range=(1, 1), min_df=min_df,
                        max_features=max_features, sublinear_tf=True)
    return (v.fit_transform([text[i] for i in tr]).toarray(),
            v.transform([text[i] for i in te]).toarray())


def f_tfidf(tr, te):      return _tfidf(tr, te)


def f_tfidf_triv(tr, te):
    """The union baseline: lexical content AND surface size. More capacity for
    the baseline is the conservative direction, per emobank_baseline.py."""
    a, b = _tfidf(tr, te)
    return np.hstack([a, Xtriv[tr]]), np.hstack([b, Xtriv[te]])


FEATS = {"probe": f_probe, "probe_unit": f_probe_unit, "trivial": f_trivial,
         "tfidf": f_tfidf, "tfidf_trivial": f_tfidf_triv}


def splits(scheme):
    if scheme == "random":
        return list(KFold(5, shuffle=True, random_state=SEED).split(Y))
    return list(GroupKFold(n_splits=5).split(Y, groups=docs))


def fold_scores(fn, scheme):
    """Per-fold out-of-fold R^2 for each dimension."""
    out = {d: [] for d in DIMS}
    for tr, te in splits(scheme):
        Xtr, Xte = fn(tr, te)
        for j, d in enumerate(DIMS):
            m = RidgeCV(alphas=ALPHAS).fit(Xtr, Y[tr, j])
            out[d].append(r2_score(Y[te, j], m.predict(Xte)))
    return {d: np.array(v) for d, v in out.items()}


def paired(a, b):
    """Paired-over-folds difference a - b. 5 folds -> t(4) = 2.776."""
    r = {}
    for d in DIMS:
        diff = a[d] - b[d]
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        r[d] = {"mean_diff": float(diff.mean()), "se": float(se),
                "lo": float(diff.mean() - 2.776 * se),
                "hi": float(diff.mean() + 2.776 * se),
                "all_folds_positive": bool((diff > 0).all()),
                "per_fold": [float(x) for x in diff]}
    return r


R = {"n": n, "n_documents": int(len(set(docs))), "seed": SEED,
     "note": "Appends to, never rewrites, emobank_baseline.json. "
             "'probe_norm' is ||P a||, a magnitude proxy; ||a|| was never stored."}

# ---------------------------------------------------------------- fold scores
S = {}
for scheme in ("random", "grouped"):
    S[scheme] = {k: fold_scores(fn, scheme) for k, fn in FEATS.items()}
    print(f"\n[{scheme} folds]")
    for k, v in S[scheme].items():
        print(f"  {k:14s} V {v['V'].mean():+.3f}  A {v['A'].mean():+.3f}  "
              f"D {v['D'].mean():+.3f}")

R["cv_r2"] = {sc: {k: {d: float(v[d].mean()) for d in DIMS}
                   for k, v in S[sc].items()} for sc in S}

# ------------------------------------- CONSTRUCT: is it affect or activation size?
R["construct"] = {
    "probe_unit_vs_tfidf": {sc: paired(S[sc]["probe_unit"], S[sc]["tfidf"]) for sc in S},
    "probe_vs_probe_unit": {sc: paired(S[sc]["probe"], S[sc]["probe_unit"]) for sc in S},
}

# ----------------------------------------- CONTENT: union baseline is the honest one
R["content"] = {
    "probe_vs_tfidf_trivial": {sc: paired(S[sc]["probe"], S[sc]["tfidf_trivial"]) for sc in S},
    "probe_unit_vs_tfidf_trivial": {sc: paired(S[sc]["probe_unit"], S[sc]["tfidf_trivial"]) for sc in S},
    "level_spread_random": {d: float(S["random"]["probe"][d].mean()) for d in DIMS},
}

# --------------------------------------- DISCRIMINANT: does it reduce to nuisance?
R["discriminant"] = {
    "trivial_alone": {sc: {d: float(S[sc]["trivial"][d].mean()) for d in DIMS} for sc in S},
    "probe_vs_trivial": {sc: paired(S[sc]["probe"], S[sc]["trivial"]) for sc in S},
    "gap_declared": "50 random directions on the same activations is the sharper "
                    "control and is NOT available: only the 50 emotion projections "
                    "were stored for EmoBank, not the activations. Needs GPU.",
}

# ------------- CONVERGENT: is 'low-arousal negative' true in HUMAN coordinates?
conv = {}
for i, e in enumerate(emos):
    conv[e] = {d: float(np.corrcoef(Xp[:, i], Y[:, j])[0, 1]) for j, d in enumerate(DIMS)}
R["convergent"] = {"per_direction_corr_with_human_VAD": conv}
# The directions surviving BOTH readouts. READ from the results file, never
# typed from memory: an earlier draft of this script hardcoded a list that was
# wrong on 4 of 8. conditional_null.py writes direction_only.overlap_with_scalar.
SURV8 = json.load(open(ROOT / "results/conditional_null.json")
                  )["direction_only"]["overlap_with_scalar"]
have = [e for e in SURV8 if e in conv]
R["convergent"]["survivors_checked"] = have
R["convergent"]["survivors_missing"] = [e for e in SURV8 if e not in conv]
if have:
    R["convergent"]["survivor_mean_V"] = float(np.mean([conv[e]["V"] for e in have]))
    R["convergent"]["survivor_mean_A"] = float(np.mean([conv[e]["A"] for e in have]))
    R["convergent"]["nonsurvivor_mean_V"] = float(np.mean(
        [conv[e]["V"] for e in emos if e not in have]))
    R["convergent"]["nonsurvivor_mean_A"] = float(np.mean(
        [conv[e]["A"] for e in emos if e not in have]))

# -------------------------------------------------- CRITERION: recorded, not re-run
R["criterion"] = {
    "concurrent": "human V/A/D, this file",
    "predictive": "reward hacking; probe adds nothing to a length-only model "
                  "(T=3.3, p=0.060) -- section 4.2, already in the paper",
}

# ------------------------------- INTERNAL: does document leakage carry the margin?
R["internal"] = {
    "probe_vs_tfidf": {sc: paired(S[sc]["probe"], S[sc]["tfidf"]) for sc in S},
    "leakage_note": "136 documents, median 30 sentences each, largest 1192. "
                    "Random KFold splits same-document sentences across train/test.",
    "largest_doc_share": float(max(np.bincount(
        np.unique(docs, return_inverse=True)[1])) / n),
}

# ------------------------------- EXTERNAL: positive across document groups, or one?
# Reuses the grouped per-fold scores already computed above rather than
# refitting: S["grouped"][k][d] IS the per-fold R^2 under GroupKFold, so the
# per-group margin is a subtraction, not a second expensive pass.
gsplits = splits("grouped")
ext = {}
for d in DIMS:
    per = []
    for fi, (tr, te) in enumerate(gsplits):
        p = float(S["grouped"]["probe"][d][fi])
        b = float(S["grouped"]["tfidf_trivial"][d][fi])
        per.append({"n_test": int(len(te)), "n_docs": int(len(set(docs[te]))),
                    "probe": p, "baseline": b, "margin": p - b})
    ext[d] = {"folds": per,
              "all_groups_positive": bool(all(x["margin"] > 0 for x in per)),
              "min_margin": float(min(x["margin"] for x in per))}
R["external"] = {"per_document_group": ext}

# ----------------------- ECOLOGICAL: is EmoBank the setting the probe is used in?
#
# READOUT MISMATCH -- established by reading the two writers, not assumed:
#   EmoBank  scripts/run_emobank_validation.py:136
#            dot(residual, v) / (||residual|| * ||v||)   -> a TRUE COSINE,
#            taken at the LAST TOKEN of a raw sentence, no chat template.
#   Trials   scripts/phase2_steering.py:212
#            dot(mean_act, v) / ||v||                    -> a SCALAR PROJECTION,
#            mean over response tokens from offset 50 on, with chat template.
# (src/experiments.py:82 does normalize, but it is not the writer for these
# trial files; phase2_steering.py is.)
#
# So sections 4.1 and 4.2-4.5 do not measure the same quantity, and comparing
# their raw values is meaningless: 1719/6000 trial values exceed |1|, which no
# cosine can. An earlier draft of this block compared them raw and reported a
# "15 SD domain shift". That number was a readout artifact and is not reported.
#
# The direction-only readout is the one quantity both sides CAN be put in:
#   EmoBank c = (P r)/||r||  ->  c/||c|| = P r / ||P r||
#   Trials   p = P a         ->  p/||p|| = P a / ||P a||
# identical functional form, each cancelling its own scale factor exactly. All
# probe comparisons below are therefore made on unit-normalized 50-vectors.
eco = {"readout_mismatch": {
    "emobank": "cosine, last token, raw sentence (run_emobank_validation.py:136)",
    "trials": "scalar projection, mean from token 50, chat template "
              "(phase2_steering.py:212)",
    "trial_values_exceeding_abs_1": None,
    "commensurable_readout": "unit-normalized 50-vector; cancels each side's "
                             "scale factor exactly",
}}

eco["emobank_chars"] = {"mean": float(n_chars.mean()),
                        "median": float(np.median(n_chars)),
                        "p5": float(np.percentile(n_chars, 5)),
                        "p95": float(np.percentile(n_chars, 95))}

# The trial set is built EXACTLY as paper_numbers.py:82-87 builds it -- the
# _judged files, filtered on judge_classification -- so this compares against
# the same 120 trials the paper's inference lives on. Reading the raw
# (unjudged) files instead gives 200 trials: a different population.
def _load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


P4 = ROOT / "results/phase4/llama70b"
P2 = ROOT / "results/phase2"
trials = _load(P4 / "extended_unsteered_judged.jsonl")
trials += [t for t in _load(P2 / "task_a_judged.jsonl")
           if float(t.get("strength", 0)) == 0.0]
trials = [t for t in trials
          if t.get("judge_classification") in ("SHORTCUT", "LEGITIMATE")
          and t.get("emotion_probes")]
eco["n_trials_matched_paper"] = len(trials)

Xt = np.array([[t["emotion_probes"][e] for e in emos] for t in trials], float)
eco["readout_mismatch"]["trial_values_exceeding_abs_1"] = int((np.abs(Xt) > 1).sum())
eco["readout_mismatch"]["trial_values_total"] = int(Xt.size)
eco["readout_mismatch"]["emobank_within_pm1"] = bool((np.abs(Xp) <= 1).all())

# --- length: no readout involved, so this comparison is valid as measured ----
r_ = np.array([len(t.get("response", "")) for t in trials], float)
eco["trial_response_chars"] = {"n": len(trials), "mean": float(r_.mean()),
                               "median": float(np.median(r_)),
                               "p5": float(np.percentile(r_, 5)),
                               "p95": float(np.percentile(r_, 95))}
eco["median_length_ratio"] = float(np.median(r_) / np.median(n_chars))
eco["frac_emobank_inside_trial_p5_p95"] = float(
    ((n_chars >= np.percentile(r_, 5)) & (n_chars <= np.percentile(r_, 95))).mean())

# --- probe geometry on the commensurable (unit-normalized) readout ----------
Ut = Xt / np.linalg.norm(Xt, axis=1, keepdims=True)
Ue = Xp_unit
me, mt = Ue.mean(0), Ut.mean(0)
eco["direction_only"] = {
    "cos_between_mean_directions": float(
        me @ mt / (np.linalg.norm(me) * np.linalg.norm(mt))),
    "emobank_mean_resultant_length": float(np.linalg.norm(me)),
    "trial_mean_resultant_length": float(np.linalg.norm(mt)),
}
if "desperate" in emos:
    k = emos.index("desperate")
    e_, t_ = Ue[:, k], Ut[:, k]
    eco["direction_only"]["desperate"] = {
        "emobank_mean": float(e_.mean()), "emobank_sd": float(e_.std(ddof=1)),
        "trial_mean": float(t_.mean()), "trial_sd": float(t_.std(ddof=1)),
        "standardized_gap_in_emobank_sd": float(
            (t_.mean() - e_.mean()) / e_.std(ddof=1)),
        "trial_frac_inside_emobank_p5_p95": float(
            ((t_ >= np.percentile(e_, 5)) & (t_ <= np.percentile(e_, 95))).mean()),
    }
R["ecological"] = eco

OUT.write_text(json.dumps(R, indent=1, sort_keys=True))
print(f"\nwrote {OUT.relative_to(ROOT)}")
