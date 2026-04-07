#!/usr/bin/env python3
"""
Phase 4: Control experiments.

  Control 1: Random direction steering (specificity check)
  Control 2: Text injection (text vs activation disentanglement)
  Control 3: Layer sweep (layer specificity)

Usage:
    python scripts/04_run_controls.py --model llama-70b
    python scripts/04_run_controls.py --model llama-70b --control random
    python scripts/04_run_controls.py --model llama-70b --control injection
    python scripts/04_run_controls.py --model llama-70b --control layers
"""

import argparse
import sys
import json
import numpy as np
import torch
from pathlib import Path
from dataclasses import asdict
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODELS, ensure_dirs, DATA_DIR, RESULTS_DIR,
    Phase2Settings, TASK_A_VARIANTS,
)
from src.model import ModelWrapper, SteeringConfig
from src.vectors import load_emotion_vectors
from src.experiments import (
    run_random_direction_control,
    run_text_injection_control,
    run_steering_experiment,
    code_coding_outcome,
    TrialResult,
)

from loguru import logger


def run_layer_sweep(
    model: ModelWrapper,
    model_config,
    phase1_dir: Path,
    output_dir: Path,
    settings: Phase2Settings,
):
    """
    Control 3: Run steering at different layers to check layer specificity.
    Prediction: mid-to-late layers have strongest behavioral effects.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = model_config.all_analysis_layers
    tasks = TASK_A_VARIANTS[:2]
    emotion = "desperate"

    results_by_layer = {}

    for layer_idx in layers:
        logger.info(f"\n--- Layer {layer_idx} ---")

        try:
            vectors = load_emotion_vectors(layer_idx, phase1_dir / "vectors")
        except FileNotFoundError:
            logger.warning(f"No vectors for layer {layer_idx}, skipping")
            continue

        if emotion not in vectors:
            logger.warning(f"'{emotion}' not found at layer {layer_idx}")
            continue

        vec = torch.from_numpy(vectors[emotion]).float()

        # Compute residual norm at this layer
        calibration_texts = [
            "The weather is nice today.",
            "Please help me with this task.",
        ]
        norms = model.compute_residual_norms(calibration_texts, [layer_idx])
        norm = norms[layer_idx]

        layer_results = []
        for task in tasks:
            try:
                messages = [{"role": "user", "content": task["prompt"]}]
                if task.get("system"):
                    messages.insert(0, {"role": "system", "content": task["system"]})
                full_prompt = model.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                full_prompt = task["prompt"]

            for strength in [0.0, 0.05, -0.05]:
                for trial_idx in range(settings.trials_per_condition):
                    steering_configs = []
                    if abs(strength) > 1e-6:
                        steering_configs.append(SteeringConfig(
                            vector=vec, layer_idx=layer_idx,
                            strength=strength, residual_norm=norm,
                        ))
                    output = model.generate_steered(
                        prompt=full_prompt,
                        steering_configs=steering_configs,
                        max_new_tokens=settings.max_new_tokens,
                        temperature=settings.temperature,
                        top_p=settings.top_p,
                    )
                    outcome = code_coding_outcome(output["text"])
                    layer_results.append({
                        "layer": layer_idx,
                        "task_id": task["id"],
                        "strength": strength,
                        "shortcut": outcome["shortcut"],
                        "trial_idx": trial_idx,
                    })

        results_by_layer[layer_idx] = layer_results

        # Summarize this layer
        rates = defaultdict(list)
        for r in layer_results:
            rates[r["strength"]].append(r["shortcut"])
        for s, outcomes in sorted(rates.items()):
            rate = sum(outcomes) / len(outcomes)
            logger.info(f"  Strength {s:+.3f}: shortcut rate = {rate:.3f} (n={len(outcomes)})")

        # Save after each layer
        all_results = []
        for lr in results_by_layer.values():
            all_results.extend(lr)
        with open(output_dir / "layer_sweep_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Checkpoint: saved layer sweep results through layer {layer_idx}")

    return results_by_layer


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Controls")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument(
        "--control", type=str, default="all",
        choices=["all", "random", "injection", "layers"],
        help="Which control to run (default: all)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer trials"
    )
    args = parser.parse_args()

    ensure_dirs()
    model_config = MODELS[args.model]
    settings = Phase2Settings()
    if args.quick:
        settings.trials_per_condition = 10

    phase1_dir = DATA_DIR / "phase1" / model_config.short_name
    controls_dir = RESULTS_DIR / "phase4" / model_config.short_name
    controls_dir.mkdir(parents=True, exist_ok=True)

    model = ModelWrapper(model_config.name, cache_dir=args.cache_dir)
    primary_layer = model_config.mid_late_layer
    emotion_vectors = load_emotion_vectors(primary_layer, phase1_dir / "vectors")

    # Load residual norms
    norms_file = phase1_dir / "residual_norms.npz"
    if norms_file.exists():
        loaded = np.load(norms_file)
        residual_norms = {int(k.split("_")[1]): float(loaded[k]) for k in loaded.files}
    else:
        residual_norms = model.compute_residual_norms(
            ["Test sentence."], [primary_layer]
        )

    # ----------------------------------------------------------------
    # Control 1: Random directions
    # ----------------------------------------------------------------
    if args.control in ["all", "random"]:
        logger.info(f"\n{'='*60}")
        logger.info("CONTROL 1: Random Direction Steering")
        logger.info(f"{'='*60}")

        random_results = run_random_direction_control(
            model, emotion_vectors, residual_norms, primary_layer,
            n_random_dirs=10,
            trials_per_dir=settings.trials_per_condition // 2,
            settings=settings,
            output_dir=controls_dir / "random_directions",
        )

        # Summarize
        shortcut_rates = defaultdict(list)
        for r in random_results:
            shortcut_rates[r.strength].append(r.outcome.get("shortcut", False))
        for s, outcomes in sorted(shortcut_rates.items()):
            rate = sum(outcomes) / len(outcomes)
            logger.info(f"  Random steering {s:+.3f}: shortcut rate = {rate:.3f} (n={len(outcomes)})")

    # ----------------------------------------------------------------
    # Control 2: Text injection
    # ----------------------------------------------------------------
    if args.control in ["all", "injection"]:
        logger.info(f"\n{'='*60}")
        logger.info("CONTROL 2: Text Injection")
        logger.info(f"{'='*60}")

        injection_results = run_text_injection_control(
            model, emotion_vectors, primary_layer,
            trials_per_condition=settings.trials_per_condition,
            output_dir=controls_dir / "text_injection",
        )

        # Summarize
        inject_rates = defaultdict(list)
        for r in injection_results:
            if "shortcut" in r.outcome:
                inject_rates[r.emotion].append(r.outcome["shortcut"])
            elif "sycophantic" in r.outcome:
                inject_rates[r.emotion].append(r.outcome["sycophantic"])
        for inject_type, outcomes in sorted(inject_rates.items()):
            rate = sum(outcomes) / len(outcomes)
            logger.info(f"  {inject_type}: misaligned rate = {rate:.3f} (n={len(outcomes)})")

    # ----------------------------------------------------------------
    # Control 3: Layer sweep
    # ----------------------------------------------------------------
    if args.control in ["all", "layers"]:
        logger.info(f"\n{'='*60}")
        logger.info("CONTROL 3: Layer Sweep")
        logger.info(f"{'='*60}")

        layer_results = run_layer_sweep(
            model, model_config, phase1_dir,
            controls_dir / "layer_sweep",
            settings,
        )

    logger.info(f"\nAll controls complete. Results in {controls_dir}")
    model.cleanup()


if __name__ == "__main__":
    main()
