#!/usr/bin/env python3
"""
Trivial-text baseline for section 4.1, added in response to review 2026-08-20.

The manuscript argues that a probe should have to beat a trivial feature of the
output before it is called a mechanism (section 6), then reports EmoBank CV R^2 =
0.377 for the 50-d probe with nothing to compare it against. This supplies the
missing comparison: unigram TF-IDF on the same sentences, same folds, same
estimator family, predicting the same human V/A/D ratings.

Protocol is copied from scripts/run_emobank_validation.py:210-221 -- RidgeCV over
alphas [0.1, 1, 10, 100], KFold(5, shuffle=True, random_state=20260510) -- so the
only thing that differs between the two rows is the feature set.

Two TF-IDF rows are reported. The capped row (2000 unigrams, min_df=2) is the
original. The uncapped row (8000 unigrams, min_df=1) is the one the manuscript
quotes, because capping the baseline makes it WEAKER and therefore flatters the
probe; an earlier version of this file called the cap "conservative", which had
the direction of the bias backwards. A baseline is treated conservatively by
being given more capacity, not less. On valence the cap costs the baseline about
0.08 R^2, which is more than half the probe's apparent margin.

Analysis-only. Writes results/emobank_baseline.json.  Runtime ~10 min.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
TRIALS = ROOT / "results/emobank/llama70b/trials.jsonl"
OUT = ROOT / "results/emobank_baseline.json"
DIMS = ["V", "A", "D"]

rows = [json.loads(l) for l in open(TRIALS) if l.strip()]
text = [r["text"] for r in rows]
Y = np.array([[r[d] for d in DIMS] for r in rows], float)
emos = sorted(rows[0]["emotion_probes"])
Xp = np.array([[r["emotion_probes"][e] for e in emos] for r in rows], float)
print(f"n = {len(rows)} sentences, {len(emos)} probe dimensions")

kf = KFold(n_splits=5, shuffle=True, random_state=20260510)


def cv_r2(make_features):
    out = {d: [] for d in DIMS}
    for tr, te in kf.split(Y):
        Xtr, Xte = make_features(tr, te)
        for j, d in enumerate(DIMS):
            m = RidgeCV(alphas=[0.1, 1, 10, 100]).fit(Xtr, Y[tr, j])
            out[d].append(r2_score(Y[te, j], m.predict(Xte)))
    return {d: float(np.mean(v)) for d, v in out.items()}


def probe_features(tr, te):
    return Xp[tr], Xp[te]


def make_tfidf(max_features, min_df):
    """Fitted inside the fold: the vocabulary never sees held-out sentences."""
    def f(tr, te):
        v = TfidfVectorizer(lowercase=True, ngram_range=(1, 1), min_df=min_df,
                            max_features=max_features, sublinear_tf=True)
        return (v.fit_transform([text[i] for i in tr]).toarray(),
                v.transform([text[i] for i in te]).toarray())
    return f


probe = cv_r2(probe_features)
tfidf = cv_r2(make_tfidf(2000, 2))
tfidf_unc = cv_r2(make_tfidf(8000, 1))
print(f"  probe (50-d)          V {probe['V']:.3f}  A {probe['A']:.3f}  D {probe['D']:.3f}")
print(f"  TF-IDF 2000 (capped)  V {tfidf['V']:.3f}  A {tfidf['A']:.3f}  D {tfidf['D']:.3f}")
print(f"  TF-IDF 8000 (uncapped) V {tfidf_unc['V']:.3f}  A {tfidf_unc['A']:.3f}"
      f"  D {tfidf_unc['D']:.3f}   <- quoted in the manuscript")
print("  capping the baseline flatters the probe; the uncapped row is the fair one")

R = {"n": len(rows), "probe_cv_r2": probe, "tfidf_cv_r2": tfidf,
     "tfidf_uncapped_cv_r2": tfidf_unc,
     "delta": {d: probe[d] - tfidf[d] for d in DIMS},
     "delta_uncapped": {d: probe[d] - tfidf_unc[d] for d in DIMS},
     "protocol": "RidgeCV alphas [0.1,1,10,100], KFold(5, shuffle, seed 20260510); "
                 "TF-IDF fitted within fold. Capped row: top-2000 unigrams, "
                 "min_df=2. Uncapped row: top-8000 unigrams, min_df=1 -- the "
                 "manuscript quotes the uncapped row because capping weakens the "
                 "baseline and so flatters the probe.",
     "probe_beats_tfidf": {d: bool(probe[d] > tfidf[d]) for d in DIMS},
     "probe_beats_tfidf_uncapped": {d: bool(probe[d] > tfidf_unc[d]) for d in DIMS}}
OUT.write_text(json.dumps(R, indent=1, sort_keys=True))
print(f"wrote {OUT.relative_to(ROOT)}")

# ---------------------------------------------------------------------------
# Is the probe's margin over the baseline actually distinguishable from zero?
# Table 1's one "Supported" verdict rests on it, and a paper about untested
# margins should not leave its own untested. Paired over CV folds.
# ---------------------------------------------------------------------------
def fold_scores(make_features):
    out = {d: [] for d in DIMS}
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=20260510).split(Y):
        Xtr, Xte = make_features(tr, te)
        for j, d in enumerate(DIMS):
            m = RidgeCV(alphas=[0.1, 1, 10, 100]).fit(Xtr, Y[tr, j])
            out[d].append(r2_score(Y[te, j], m.predict(Xte)))
    return out


_pf = fold_scores(probe_features)
_bf = fold_scores(make_tfidf(8000, 1))
paired = {}
print("\n  paired over 5 CV folds (probe - uncapped TF-IDF):")
for d in DIMS:
    diff = np.array(_pf[d]) - np.array(_bf[d])
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    paired[d] = {"mean_diff": float(diff.mean()), "se": float(se),
                 "lo": float(diff.mean() - 2.776 * se), "hi": float(diff.mean() + 2.776 * se),
                 "all_folds_positive": bool((diff > 0).all())}
    print(f"    {d}  diff {diff.mean():+.3f}  95% CI [{paired[d]['lo']:+.3f}, "
          f"{paired[d]['hi']:+.3f}]  all folds positive: {paired[d]['all_folds_positive']}")
R = json.loads(OUT.read_text())
R["paired_fold_diff_vs_uncapped"] = paired
OUT.write_text(json.dumps(R, indent=1, sort_keys=True))
