#!/usr/bin/env python3
"""
Judge symmetry / confound check — addresses red-team finding M5.

The judge prompt is symmetric on inspection (anchored at neutral, no
emotion priming, doesn't see the steering condition). This script
verifies symmetry empirically using existing measurements:

  Confound matrix C[i, j] = corr(V_text_dim_i, V_internal_emotion_j)
  across all Phase 2 trials.

A *symmetric* judge should produce a confound matrix where each V_text
dimension lights up most for the V_internal probe it semantically maps
to (urgency↔desperate, composure↔calm, valence↔happy/loving, ...).
An *asymmetric / desperate-biased* judge would light up urgency for ALL
emotions equally, or fail to differentiate.

Decision rule (preregistered for the appendix):
  - If urgency.corr(desperate) > urgency.corr(calm) by >= 0.10
    AND composure.corr(calm) > composure.corr(desperate) by >= 0.10
    THEN judge symmetry is supported.
  - If both checks fail OR urgency.corr is uniform across emotions,
    judge is biased and V_text values must be reinterpreted.

Output: results/phase3/llama70b/judge_symmetry.json + brief stdout summary.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR, MODELS

VTEXT_DIMS = [
    "valence", "arousal", "dominance", "urgency",
    "composure", "frustration", "hedging", "self_interruption",
]
EMOTIONS = ["desperate", "calm", "angry", "afraid", "happy", "loving"]

# Expected semantic mappings (predicted strongest pair per V_text dim)
EXPECTED = {
    "urgency": "desperate",
    "composure": "calm",
    "valence": "happy",
    "frustration": "angry",
    "arousal": "afraid",   # arousal is bipolar — afraid is plausible match
    "hedging": None,       # no strong prior
    "self_interruption": None,
    "dominance": None,
}


def load_measurements(path: Path) -> pd.DataFrame:
    with open(path) as f:
        rows = json.load(f)
    return pd.DataFrame(rows)


def confound_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=VTEXT_DIMS, columns=EMOTIONS, dtype=float)
    for d in VTEXT_DIMS:
        col_t = f"v_text_judge_{d}"
        if col_t not in df:
            continue
        for e in EMOTIONS:
            col_e = f"v_internal_{e}"
            if col_e not in df:
                continue
            sub = df[[col_t, col_e]].dropna()
            if len(sub) < 10 or sub[col_t].std() == 0 or sub[col_e].std() == 0:
                out.loc[d, e] = float("nan")
            else:
                out.loc[d, e] = float(sub[col_t].corr(sub[col_e]))
    return out


def symmetry_audit(C: pd.DataFrame) -> dict:
    """A symmetric judge should produce a coherent signed pattern. We test
    two things per V_text dimension that has an expected polarity:
      (a) sign of correlation with the expected emotion is correct, and
      (b) correlation with the *opposite* emotion has the opposite sign.
    Argmax-by-magnitude is NOT the right test, because two emotions can
    legitimately have similar |r| with opposite sign (e.g., loving and
    desperate both correlate strongly with urgency, in opposite directions —
    this is exactly what symmetry predicts)."""
    audit = {}
    # opposite-emotion pairs (semantic anti-pairs)
    OPPOSITES = {"desperate": "calm", "calm": "desperate",
                 "happy": "angry", "angry": "happy",
                 "loving": "afraid", "afraid": "loving"}
    for vd, expected_emo in EXPECTED.items():
        if expected_emo is None:
            continue
        if expected_emo not in C.columns:
            continue
        row = C.loc[vd].dropna()
        if expected_emo not in row.index:
            continue
        target = float(row[expected_emo])
        opposite = OPPOSITES.get(expected_emo)
        opposite_corr = float(row[opposite]) if (opposite and opposite in row.index) else float("nan")
        # Each (V_text_dim, expected_emotion) pair is constructed so the
        # expected emotion is the one that should *increase* the V_text
        # dim (e.g., calm → high composure, desperate → high urgency,
        # happy → high valence). So the predicted sign is always +1.
        expected_sign = +1
        sign_correct = (np.sign(target) == expected_sign) and abs(target) >= 0.10
        opposite_sign_correct = (
            not np.isnan(opposite_corr)
            and np.sign(opposite_corr) == -expected_sign
            and abs(opposite_corr) >= 0.10
        )
        audit[vd] = {
            "expected_emotion": expected_emo,
            "expected_sign": expected_sign,
            "corr_with_expected": target,
            "opposite_emotion": opposite,
            "corr_with_opposite": opposite_corr,
            "expected_sign_correct": bool(sign_correct),
            "opposite_sign_correct": bool(opposite_sign_correct),
            "passes": bool(sign_correct and opposite_sign_correct),
        }

    # Headline checks per the docstring decision rule
    headline = {
        "urgency_desp_minus_calm": float(C.loc["urgency", "desperate"]) - float(C.loc["urgency", "calm"]),
        "composure_calm_minus_desp": float(C.loc["composure", "calm"]) - float(C.loc["composure", "desperate"]),
    }
    headline["urgency_pass"] = headline["urgency_desp_minus_calm"] >= 0.10
    headline["composure_pass"] = headline["composure_calm_minus_desp"] >= 0.10
    headline["overall_pass"] = bool(headline["urgency_pass"] and headline["composure_pass"])

    return {"per_dim": audit, "headline": headline}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    args = ap.parse_args()

    short = MODELS[args.model].short_name
    in_path = RESULTS_DIR / "phase3" / short / "faithfulness_measurements.json"
    out_path = RESULTS_DIR / "phase3" / short / "judge_symmetry.json"

    df = load_measurements(in_path)
    C = confound_matrix(df)
    audit = symmetry_audit(C)

    report = {
        "n_rows_used": int(len(df)),
        "confound_matrix": {d: {e: (None if pd.isna(v) else float(v))
                                for e, v in row.items()}
                            for d, row in C.iterrows()},
        "symmetry_audit": audit,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 72)
    print("JUDGE SYMMETRY CHECK")
    print("=" * 72)
    print("Confound matrix (Pearson corr between V_text dim and V_internal probe):")
    print(C.to_string(float_format=lambda v: f"{v:+.3f}"))
    print()
    print("Headline:")
    for k, v in audit["headline"].items():
        print(f"  {k}: {v}")
    print()
    print("Per-dimension signed-pattern check:")
    for d, a in audit["per_dim"].items():
        flag = "OK" if a["passes"] else "FAIL"
        print(f"  [{flag}] {d}: r({a['expected_emotion']})={a['corr_with_expected']:+.3f} "
              f"(want sign {a['expected_sign']:+d}), "
              f"r({a['opposite_emotion']})={a['corr_with_opposite']:+.3f}")
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
