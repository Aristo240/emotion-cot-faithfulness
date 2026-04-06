#!/usr/bin/env python3
"""
Phase 1: Full emotion vector extraction and validation.

50 emotions × 100 topics × 12 stories = 60,000 stories.
Plus 500 neutral dialogues for PCA denoising.
Full validation battery with go/no-go decision.

Usage:
    python scripts/01_run_phase1.py --model llama-70b
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS, ensure_dirs, DATA_DIR, RESULTS_DIR, Phase1Settings
from src.model import ModelWrapper
from src.generate import generate_stories, generate_neutral_dialogues
from src.vectors import (
    extract_story_activations,
    extract_neutral_activations,
    compute_emotion_vectors,
)
from src.validate import run_full_validation
from src.analysis import plot_cosine_similarity_matrix, plot_pca_scatter

from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Full emotion vector extraction")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="Skip story generation (use cached stories)"
    )
    args = parser.parse_args()

    ensure_dirs()
    settings = Phase1Settings()
    model_config = MODELS[args.model]

    phase1_dir = DATA_DIR / "phase1" / model_config.short_name
    phase1_dir.mkdir(parents=True, exist_ok=True)
    results_dir = RESULTS_DIR / "phase1" / model_config.short_name
    results_dir.mkdir(parents=True, exist_ok=True)

    model = ModelWrapper(model_config.name, cache_dir=args.cache_dir)

    # Use mid-late layer as primary + a few others for layer sweep
    primary_layer = model_config.mid_late_layer
    all_layers = sorted(set(model_config.all_analysis_layers + [primary_layer]))
    logger.info(f"Analysis layers: {all_layers}, primary: {primary_layer}")

    # ----------------------------------------------------------------
    # Step 1: Generate stories (largest compute step)
    # ----------------------------------------------------------------
    if not args.skip_generation:
        logger.info(f"Generating {len(settings.emotions)} emotions × {len(settings.topics)} topics × {settings.stories_per_topic} stories")
        stories = generate_stories(
            model,
            emotions=settings.emotions,
            topics=settings.topics,
            stories_per_topic=settings.stories_per_topic,
            output_dir=phase1_dir / "stories",
        )

        logger.info(f"Generating {len(settings.topics) * settings.neutral_dialogues_per_topic} neutral dialogues")
        neutral = generate_neutral_dialogues(
            model,
            topics=settings.topics,
            dialogues_per_topic=settings.neutral_dialogues_per_topic,
            output_dir=phase1_dir / "neutral",
        )
    else:
        # Load from cache
        stories = {}
        for emotion in settings.emotions:
            fpath = phase1_dir / "stories" / f"{emotion.replace(' ', '_')}.json"
            if fpath.exists():
                with open(fpath) as f:
                    stories[emotion] = json.load(f)
        with open(phase1_dir / "neutral" / "neutral_dialogues.json") as f:
            neutral = json.load(f)

    total_stories = sum(len(s) for s in stories.values())
    logger.info(f"Total stories: {total_stories}, Neutral dialogues: {len(neutral)}")

    # ----------------------------------------------------------------
    # Step 2: Extract activations
    # ----------------------------------------------------------------
    logger.info("Extracting story activations...")
    emotion_means = extract_story_activations(
        model, stories, all_layers, token_offset=settings.token_offset,
        output_dir=phase1_dir / "activations",
    )

    logger.info("Extracting neutral activations...")
    neutral_acts = extract_neutral_activations(
        model, neutral, all_layers, token_offset=settings.token_offset,
        output_dir=phase1_dir / "activations",
    )

    # ----------------------------------------------------------------
    # Step 3: Compute emotion vectors
    # ----------------------------------------------------------------
    logger.info("Computing emotion vectors...")
    vectors = compute_emotion_vectors(
        emotion_means, neutral_acts, all_layers,
        variance_threshold=settings.neutral_pca_variance_threshold,
        output_dir=phase1_dir / "vectors",
    )

    # ----------------------------------------------------------------
    # Step 4: Compute residual norms for steering calibration
    # ----------------------------------------------------------------
    logger.info("Computing residual stream norms for steering calibration...")
    calibration_texts = [d["text"] for d in neutral[:50]]
    residual_norms = model.compute_residual_norms(calibration_texts, all_layers)
    np.savez(
        phase1_dir / "residual_norms.npz",
        **{f"layer_{k}": np.array([v]) for k, v in residual_norms.items()}
    )
    for layer_idx, norm in residual_norms.items():
        logger.info(f"  Layer {layer_idx}: mean residual norm = {norm:.1f}")

    # ----------------------------------------------------------------
    # Step 5: Full validation battery
    # ----------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("FULL VALIDATION BATTERY")
    logger.info(f"{'='*60}")

    primary_vectors = vectors[primary_layer]
    validation_results = run_full_validation(
        model, primary_vectors, stories, primary_layer, settings.token_offset
    )

    # Save results
    val_summary = []
    for r in validation_results:
        val_summary.append({
            "test": r.test_name,
            "passed": r.passed,
            "value": r.value,
            "threshold": r.threshold,
            "details": r.details,
        })
    with open(results_dir / "validation_results.json", "w") as f:
        json.dump(val_summary, f, indent=2)

    # Plots
    from src.validate import validate_cosine_similarity, validate_pca_vs_human
    sim_matrix, emo_list, _ = validate_cosine_similarity(primary_vectors)
    plot_cosine_similarity_matrix(sim_matrix, emo_list, results_dir / "cosine_similarity.png")

    _, _, projections = validate_pca_vs_human(primary_vectors)
    plot_pca_scatter(projections, sorted(primary_vectors.keys()), results_dir / "pca_scatter.png")

    # Final decision
    all_passed = all(r.passed for r in validation_results)
    critical_passed = all(
        r.passed for r in validation_results
        if r.test_name in ["cosine_similarity_opposites", "logit_lens_match"]
    )

    if critical_passed:
        logger.info("\n✓ Phase 1 PASSED. Proceed to Phase 2.")
        logger.info(f"  python scripts/02_run_phase2.py --model {args.model}")
    else:
        logger.warning("\n✗ Phase 1 FAILED critical tests. Review results before proceeding.")

    model.cleanup()


if __name__ == "__main__":
    main()
