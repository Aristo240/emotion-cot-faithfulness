#!/usr/bin/env python3
"""
Phase 3: Core faithfulness analysis.

Takes Phase 2 trial results and:
  1. Extracts chain-of-thought from each response
  2. Measures V_text (text-expressed emotion) via lexical features and LLM judge
  3. Runs all four core analyses:
     - Analysis 1: Faithfulness correlation (V_internal vs V_text)
     - Analysis 2: Predictive comparison (which predicts behavior better?)
     - Analysis 3: Dissociation case identification
     - Analysis 4: Natural (unsteered) prediction (THE KEY RESULT)

Usage:
    python scripts/03_run_phase3.py --model llama-70b
    python scripts/03_run_phase3.py --model llama-70b --skip-judge  # Skip LLM judge (faster)
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS, ensure_dirs, DATA_DIR, RESULTS_DIR, Phase3Settings
from src.experiments import TrialResult, measure_faithfulness, extract_cot, measure_v_text_lexical
from src.analysis import generate_full_report

from loguru import logger


def load_trials(trials_dir: Path) -> list:
    """Load all trial results from Phase 2 (TrialResult-format JSON)."""
    trials = []
    for f in sorted(trials_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
            for d in data:
                trials.append(TrialResult(**d))
    return trials


def _outcome_from_jsonl(d: dict, outcome_key: str) -> bool:
    """Derive the binary outcome from a jsonl row, preferring judge over regex."""
    judge = (d.get("judge_classification") or "").upper()
    cls = (d.get("classification") or "").lower()
    if outcome_key == "shortcut":
        if judge:
            return judge == "SHORTCUT"
        return cls == "hack"
    if outcome_key == "sycophantic":
        if judge:
            return judge in ("SYCOPHANTIC", "SYCOPHANCY")
        return cls == "sycophantic"
    return False


def build_measurements_from_jsonl(phase2_dir: Path, settings: Phase3Settings) -> list:
    """Build Phase 3 measurement dicts directly from the phase2 jsonl outputs
    produced by scripts/phase2_steering.py + the LLM judge pass.

    This bypasses the TrialResult pipeline (which would require re-running the
    model) because V_internal (emotion_probes), V_text judge ratings
    (vtext_ratings), and outcome classifications are all already persisted.
    """
    sources = [
        ("task_a_judged.jsonl", "shortcut"),
        ("task_a.jsonl", "shortcut"),
        ("task_b_judged.jsonl", "sycophantic"),
        ("task_b.jsonl", "sycophantic"),
    ]
    measurements = []
    seen_keys = set()
    for fname, outcome_key in sources:
        fpath = phase2_dir / fname
        if not fpath.exists():
            continue
        with open(fpath) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                # Prefer judged rows when both exist for the same trial.
                key = d.get("key") or f"{d.get('task_id')}_{d.get('emotion')}_{d.get('strength')}_{d.get('rollout')}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                probes = d.get("emotion_probes") or {}
                vtext = d.get("vtext_ratings") or {}
                vtext_std = d.get("vtext_std") or {}
                response = d.get("response", "")
                cot = extract_cot(response)
                v_text_lex = measure_v_text_lexical(
                    cot, settings.desperate_keywords, settings.calm_keywords
                )

                m = {
                    "task_id": d.get("task_id"),
                    "emotion": d.get("emotion"),
                    "strength": float(d.get("strength", 0.0)),
                    "trial_idx": int(d.get("rollout", 0)),
                    "outcome": bool(_outcome_from_jsonl(d, outcome_key)),
                    "outcome_key": outcome_key,
                    "v_internal_desperate": float(probes.get("desperate", 0.0)),
                    "v_internal_calm": float(probes.get("calm", 0.0)),
                    "v_internal_angry": float(probes.get("angry", 0.0)),
                    "v_internal_afraid": float(probes.get("afraid", 0.0)),
                    "v_internal_happy": float(probes.get("happy", 0.0)),
                    "v_internal_loving": float(probes.get("loving", 0.0)),
                    **{f"v_text_lex_{k}": v for k, v in v_text_lex.items()},
                    **{f"v_text_judge_{k}": float(v) for k, v in vtext.items()
                       if isinstance(v, (int, float))},
                    **{f"v_text_judge_std_{k}": float(v) for k, v in vtext_std.items()
                       if isinstance(v, (int, float))},
                    "cot_text": cot[:500],
                }
                measurements.append(m)
    return measurements


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Faithfulness analysis")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument(
        "--skip-judge", action="store_true",
        help="Skip LLM judge V_text measurement (use lexical only)"
    )
    parser.add_argument(
        "--analysis-only", action="store_true",
        help="Skip measurement, run analysis on existing data"
    )
    args = parser.parse_args()

    ensure_dirs()
    model_config = MODELS[args.model]
    settings = Phase3Settings()
    if args.skip_judge:
        settings.judge_repeats = 0

    # Phase 2 was run via scripts/phase2_steering.py, which writes flat jsonl
    # files directly under results/phase2/ (not under a per-model subdir).
    phase2_dir_model = RESULTS_DIR / "phase2" / model_config.short_name
    phase2_dir_flat = RESULTS_DIR / "phase2"
    phase3_dir = RESULTS_DIR / "phase3" / model_config.short_name
    phase3_dir.mkdir(parents=True, exist_ok=True)

    measurements_file = phase3_dir / "faithfulness_measurements.json"

    if not args.analysis_only:
        # Prefer jsonl outputs from phase2_steering.py — V_internal (emotion_probes),
        # V_text judge ratings, and outcome classifications are all already there,
        # so no model reload is needed.
        jsonl_src = None
        for d in (phase2_dir_model, phase2_dir_flat):
            if (d / "task_a_judged.jsonl").exists() or (d / "task_a.jsonl").exists():
                jsonl_src = d
                break

        if jsonl_src is not None:
            logger.info(f"Building measurements from phase2 jsonl in {jsonl_src}...")
            measurements = build_measurements_from_jsonl(jsonl_src, settings)
            logger.info(f"Built {len(measurements)} measurements from jsonl")
            if not measurements:
                logger.error("No rows parsed from phase2 jsonl — aborting.")
                return
            with open(measurements_file, "w") as f:
                json.dump(measurements, f, indent=2)
            logger.info(f"Measurements saved to {measurements_file}")
        else:
            # Fall back to the TrialResult-format pipeline (requires model load).
            from src.model import ModelWrapper
            from src.vectors import load_emotion_vectors
            phase1_dir = DATA_DIR / "phase1" / model_config.short_name
            model = ModelWrapper(model_config.name, cache_dir=args.cache_dir)
            primary_layer = model_config.mid_late_layer
            emotion_vectors = load_emotion_vectors(primary_layer, phase1_dir / "vectors")
            trials_dir = phase2_dir_model / "trials"
            logger.info(f"Loading trials from {trials_dir}...")
            trials = load_trials(trials_dir)
            logger.info(f"Loaded {len(trials)} trials")
            if not trials:
                logger.error(
                    f"No trials found in {trials_dir} and no jsonl outputs "
                    f"in {phase2_dir_flat}. Run Phase 2 first."
                )
                model.cleanup()
                return
            logger.info(f"\n{'='*60}")
            logger.info("PHASE 3: FAITHFULNESS MEASUREMENT")
            logger.info(f"{'='*60}")
            measure_faithfulness(
                trials, model, emotion_vectors, settings,
                output_dir=phase3_dir,
            )
            logger.info(f"Measurements saved to {measurements_file}")
            model.cleanup()
    else:
        logger.info(f"Loading existing measurements from {measurements_file}")

    # ----------------------------------------------------------------
    # Run analyses
    # ----------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 3: ANALYSIS")
    logger.info(f"{'='*60}")

    report = generate_full_report(
        measurements_file,
        phase3_dir / "analysis",
    )

    # ----------------------------------------------------------------
    # Summary of key findings
    # ----------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("KEY FINDINGS")
    logger.info(f"{'='*60}")

    # Analysis 2: Does V_internal predict behavior beyond V_text?
    a2 = report.get("analysis_2", {})
    if "v_internal_only" in a2 and "v_text_only" in a2:
        v_int_auc = a2["v_internal_only"]["auc"]
        v_text_auc = a2["v_text_only"]["auc"]
        combined_auc = a2.get("combined", {}).get("auc", 0)
        logger.info(f"  Predictive AUC:  V_internal={v_int_auc:.3f}  V_text={v_text_auc:.3f}  Combined={combined_auc:.3f}")

        if v_int_auc > v_text_auc + 0.02:
            logger.info("  → V_internal predicts behavior BETTER than V_text")
            logger.info("  → This supports the faithfulness gap hypothesis (H4)")
        else:
            logger.info("  → V_internal does NOT clearly outpredict V_text")

        lr = a2.get("likelihood_ratio_test", {})
        if lr.get("significant"):
            logger.info(f"  → Combined model significantly better than V_text alone (p={lr['p_value']:.4f})")

    # Analysis 4: Natural prediction
    a4 = report.get("analysis_4", {})
    if "v_internal" in a4:
        nat_auc = a4["v_internal"]["auc"]
        logger.info(f"\n  NATURAL (unsteered) prediction AUC: {nat_auc:.3f}")
        if nat_auc > 0.55:
            logger.info("  → V_internal predicts behavior even WITHOUT steering!")
            logger.info("  → THIS IS THE KEY RESULT (H5 supported)")
        else:
            logger.info("  → Natural prediction is weak (H5 not clearly supported)")

    # Analysis 3: Stealth cases
    a3 = report.get("analysis_3", {})
    if "stealth_rate" in a3:
        logger.info(f"\n  Stealth misalignment cases: {a3['total_stealth']}/{a3['total_trials']} ({a3['stealth_rate']:.1%})")

    logger.info(f"\nFull report: {phase3_dir / 'analysis' / 'analysis_report.json'}")
    logger.info(f"Next: python scripts/04_run_controls.py --model {args.model}")


if __name__ == "__main__":
    main()
