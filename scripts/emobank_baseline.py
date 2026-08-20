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

Analysis-only. Writes results/emobank_baseline.json.  Runtime ~1 min.
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


def tfidf_features(tr, te):
    """Fitted inside the fold: the vocabulary never sees held-out sentences.
    Capped at the 2000 most frequent unigrams so RidgeCV's exact LOO path stays
    tractable; this can only make the baseline weaker, i.e. it is conservative."""
    v = TfidfVectorizer(lowercase=True, ngram_range=(1, 1), min_df=2,
                        max_features=2000, sublinear_tf=True)
    return (v.fit_transform([text[i] for i in tr]).toarray(),
            v.transform([text[i] for i in te]).toarray())


probe = cv_r2(probe_features)
tfidf = cv_r2(tfidf_features)
print(f"  probe (50-d)      V {probe['V']:.3f}  A {probe['A']:.3f}  D {probe['D']:.3f}")
print(f"  unigram TF-IDF    V {tfidf['V']:.3f}  A {tfidf['A']:.3f}  D {tfidf['D']:.3f}")

R = {"n": len(rows), "probe_cv_r2": probe, "tfidf_cv_r2": tfidf,
     "delta": {d: probe[d] - tfidf[d] for d in DIMS},
     "protocol": "RidgeCV alphas [0.1,1,10,100], KFold(5, shuffle, seed 20260510); "
                 "TF-IDF (top-2000 unigrams) fitted within fold",
     "probe_beats_tfidf": {d: bool(probe[d] > tfidf[d]) for d in DIMS}}
OUT.write_text(json.dumps(R, indent=1, sort_keys=True))
print(f"wrote {OUT.relative_to(ROOT)}")
