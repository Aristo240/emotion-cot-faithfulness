#!/usr/bin/env python3
"""
Phase 2: Behavioral steering experiments.

Tests whether emotion vectors causally change behavior:
  - Coding tasks: does desperate steering increase shortcut-taking?
  - Sycophancy tasks: does positive-emotion steering increase agreement?
  - Unsafe tasks: does steering affect compliance?

Also runs control emotions (nostalgic, amused, grateful) to verify specificity.

Usage:
    python scripts/02_run_phase2.py --model llama-70b
    python scripts/02_run_phase2.py --model llama-70b --quick  # Fewer trials for testing
"""

import argparse
import sys
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS, ensure_dirs, DATA_DIR, RESULTS_DIR, Phase2Settings
from src.model import ModelWrapper
from src.vectors import load_emotion_vectors
from src.experiments import run_steering_experiment
from src.analysis import plot_steering_dose_response

from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Behavioral steering")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 10 trials per condition instead of 100"
    )
    args = parser.parse_args()

    ensure_dirs()
    model_config = MODELS[args.model]
    settings = Phase2Settings()
    if args.quick:
        settings.trials_per_condition = 10
        settings.steering_strengths = [-0.05, 0.0, 0.05]

    phase1_dir = DATA_DIR / "phase1" / model_config.short_name
    phase2_dir = RESULTS_DIR / "phase2" / model_config.short_name
    phase2_dir.mkdir(parents=True, exist_ok=True)

    model = ModelWrapper(model_config.name, cache_dir=args.cache_dir)
    primary_layer = model_config.mid_late_layer

    # Load emotion vectors and residual norms from Phase 1
    logger.info(f"Loading emotion vectors from Phase 1 (layer {primary_layer})...")
    emotion_vectors = load_emotion_vectors(primary_layer, phase1_dir / "vectors")
    logger.info(f"Loaded {len(emotion_vectors)} emotion vectors")

    # Check that required emotions exist
    required = settings.primary_emotions + settings.secondary_emotions + settings.control_emotions
    missing = [e for e in required if e not in emotion_vectors]
    if missing:
        logger.error(f"Missing emotion vectors: {missing}")
        logger.error("Ensure Phase 1 was run with these emotions in the subset.")
        return

    # Load residual norms
    norms_file = phase1_dir / "residual_norms.npz"
    if norms_file.exists():
        loaded = np.load(norms_file)
        residual_norms = {int(k.split("_")[1]): float(loaded[k]) for k in loaded.files}
    else:
        logger.warning("No residual norms found; computing from scratch...")
        calibration_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "In a world of constant change, adaptation is key.",
            "The temperature outside is 72 degrees Fahrenheit.",
        ]
        residual_norms = model.compute_residual_norms(calibration_texts, [primary_layer])

    logger.info(f"Residual norm at layer {primary_layer}: {residual_norms.get(primary_layer, 'N/A')}")

    # ----------------------------------------------------------------
    # Run steering experiments
    # ----------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 2: BEHAVIORAL STEERING EXPERIMENTS")
    logger.info(f"{'='*60}")
    logger.info(f"Emotions: primary={settings.primary_emotions}, secondary={settings.secondary_emotions}, control={settings.control_emotions}")
    logger.info(f"Strengths: {settings.steering_strengths}")
    logger.info(f"Trials per condition: {settings.trials_per_condition}")

    trials = run_steering_experiment(
        model,
        emotion_vectors,
        residual_norms,
        primary_layer,
        settings,
        output_dir=phase2_dir / "trials",
    )

    # ----------------------------------------------------------------
    # Summarize results
    # ----------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 2 RESULTS SUMMARY")
    logger.info(f"{'='*60}")

    # Group by (task_type, emotion, strength) and compute outcome rates
    summary = defaultdict(list)
    for t in trials:
        # Determine outcome
        if "shortcut" in t.outcome:
            outcome = t.outcome["shortcut"]
            task_type = "coding"
        elif "sycophantic" in t.outcome:
            outcome = t.outcome["sycophantic"]
            task_type = "sycophancy"
        elif "complied" in t.outcome:
            outcome = t.outcome["complied"]
            task_type = "unsafe"
        else:
            continue

        key = (task_type, t.emotion, t.strength)
        summary[key].append(outcome)

    logger.info(f"\n{'Task Type':<12} {'Emotion':<12} {'Strength':<10} {'Rate':<8} {'N':<5}")
    logger.info("-" * 50)

    for (task_type, emotion, strength), outcomes in sorted(summary.items()):
        rate = sum(outcomes) / len(outcomes)
        logger.info(f"{task_type:<12} {emotion:<12} {strength:<10.3f} {rate:<8.3f} {len(outcomes):<5}")

    # Check for significant steering effects
    logger.info("\n--- Steering Effect Summary ---")
    for emotion in settings.primary_emotions:
        # Compare baseline (0.0) to positive steering (+0.05)
        baseline_key = None
        steered_key = None
        for key in summary:
            if key[1] == emotion:
                if abs(key[2]) < 0.001:
                    baseline_key = key
                elif abs(key[2] - 0.05) < 0.001:
                    steered_key = key

        if baseline_key and steered_key:
            base_rate = sum(summary[baseline_key]) / len(summary[baseline_key])
            steer_rate = sum(summary[steered_key]) / len(summary[steered_key])
            diff = steer_rate - base_rate
            logger.info(f"  {emotion}: baseline={base_rate:.3f}, steered(+0.05)={steer_rate:.3f}, diff={diff:+.3f}")

    # Save summary
    summary_data = {
        str(k): {"rate": sum(v)/len(v), "n": len(v)}
        for k, v in summary.items()
    }
    with open(phase2_dir / "steering_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    # Plot
    import pandas as pd
    trial_dicts = []
    for t in trials:
        if "shortcut" in t.outcome:
            trial_dicts.append({
                "task_id": t.task_id, "emotion": t.emotion,
                "strength": t.strength, "outcome": t.outcome["shortcut"],
            })
        elif "sycophantic" in t.outcome:
            trial_dicts.append({
                "task_id": t.task_id, "emotion": t.emotion,
                "strength": t.strength, "outcome": t.outcome["sycophantic"],
            })
    if trial_dicts:
        plot_steering_dose_response(
            pd.DataFrame(trial_dicts),
            phase2_dir / "dose_response.png",
        )

    logger.info(f"\nPhase 2 complete. Results in {phase2_dir}")
    logger.info(f"Next: python scripts/03_run_phase3.py --model {args.model}")

    model.cleanup()


if __name__ == "__main__":
    main()
