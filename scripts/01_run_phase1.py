#!/usr/bin/env python3
"""
Phase 1: Full emotion vector extraction and validation.

50 emotions × 100 topics × 12 stories = 60,000 stories.
Plus 500 neutral dialogues for PCA denoising.
Full validation battery with go/no-go decision.

Two-stage inference:
  1. vLLM (tensor parallel) for fast story/dialogue generation
  2. HuggingFace (pipeline parallel) for activation extraction & validation

Usage:
    python scripts/01_run_phase1.py --model llama-70b
    python scripts/01_run_phase1.py --model llama-70b --skip-generation
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS, ensure_dirs, DATA_DIR, RESULTS_DIR, Phase1Settings
from src.model import ModelWrapper, FastGenerator
from src.generate import generate_stories, generate_neutral_dialogues
from src.vectors import (
    extract_story_activations,
    extract_neutral_activations,
    compute_emotion_vectors,
)
from src.validate import (
    run_full_validation,
    validate_cosine_similarity,
    validate_pca_vs_human,
    validate_cross_validation,
)
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

    # Use mid-late layer as primary + a few others for layer sweep
    primary_layer = model_config.mid_late_layer
    all_layers = sorted(set(model_config.all_analysis_layers + [primary_layer]))
    logger.info(f"Analysis layers: {all_layers}, primary: {primary_layer}")

    # ================================================================
    # STAGE 1: Generate stories & dialogues via vLLM (fast)
    # ================================================================
    if not args.skip_generation:
        logger.info(f"Loading {model_config.name} via vLLM for fast generation...")
        fast_gen = FastGenerator(model_config.name, cache_dir=args.cache_dir)

        logger.info(f"\n{'='*60}")
        logger.info(f"Generating {len(settings.emotions)} emotions × {len(settings.topics)} topics × {settings.stories_per_topic} stories")
        logger.info(f"{'='*60}")
        stories = generate_stories(
            fast_gen,
            emotions=settings.emotions,
            topics=settings.topics,
            stories_per_topic=settings.stories_per_topic,
            output_dir=phase1_dir / "stories",
        )

        logger.info(f"\nGenerating {len(settings.topics) * settings.neutral_dialogues_per_topic} neutral dialogues")
        neutral = generate_neutral_dialogues(
            fast_gen,
            topics=settings.topics,
            dialogues_per_topic=settings.neutral_dialogues_per_topic,
            output_dir=phase1_dir / "neutral",
        )

        # Free vLLM before loading HuggingFace
        fast_gen.cleanup()
        del fast_gen
    else:
        # Load from cache
        stories = {}
        for emotion in settings.emotions:
            fpath = phase1_dir / "stories" / f"{emotion.replace(' ', '_')}.json"
            if fpath.exists():
                with open(fpath) as f:
                    stories[emotion] = json.load(f)
        neutral_file = phase1_dir / "neutral" / "neutral_dialogues.json"
        with open(neutral_file) as f:
            neutral = json.load(f)

    total_stories = sum(len(s) for s in stories.values())
    logger.info(f"Total stories: {total_stories}, Neutral dialogues: {len(neutral)}")

    # Save generation summary incrementally
    gen_summary = {
        "model": model_config.name,
        "n_emotions": len(stories),
        "emotions": sorted(stories.keys()),
        "n_stories_total": total_stories,
        "stories_per_emotion": {e: len(s) for e, s in stories.items()},
        "n_neutral": len(neutral),
    }
    with open(results_dir / "generation_summary.json", "w") as f:
        json.dump(gen_summary, f, indent=2)
    logger.info(f"Generation summary saved to {results_dir / 'generation_summary.json'}")

    # ================================================================
    # STAGE 2: Extraction & validation via HuggingFace (needs hooks)
    # ================================================================
    logger.info(f"\nLoading {model_config.name} via HuggingFace for extraction...")
    model = ModelWrapper(model_config.name, cache_dir=args.cache_dir)

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

    # Save validation results
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
    logger.info(f"Validation results saved to {results_dir / 'validation_results.json'}")

    # ----------------------------------------------------------------
    # Step 6: KEY CHECK 1 — PCA structure (PC1≈valence, PC2≈arousal)
    # ----------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("KEY CHECK 1: PCA STRUCTURE (valence × arousal)")
    logger.info(f"{'='*60}")

    sim_matrix, emo_list, cos_result = validate_cosine_similarity(primary_vectors)
    valence_result, arousal_result, projections = validate_pca_vs_human(primary_vectors)

    pca_results = {
        "layer": primary_layer,
        "n_emotions": len(primary_vectors),
        "cosine_similarity": {
            "passed": cos_result.passed,
            "value": cos_result.value,
            "details": cos_result.details,
        },
        "pc1_valence_correlation": {
            "passed": valence_result.passed,
            "r": valence_result.value,
            "threshold": valence_result.threshold,
            "details": valence_result.details,
        },
        "pc2_arousal_correlation": {
            "passed": arousal_result.passed,
            "r": arousal_result.value,
            "threshold": arousal_result.threshold,
            "details": arousal_result.details,
        },
    }

    logger.info(f"  PC1 ~ valence: r={valence_result.value:.3f} (threshold {valence_result.threshold}) "
                f"{'PASS' if valence_result.passed else 'FAIL'}")
    logger.info(f"  PC2 ~ arousal: r={arousal_result.value:.3f} (threshold {arousal_result.threshold}) "
                f"{'PASS' if arousal_result.passed else 'FAIL'}")

    # Save PCA results
    with open(results_dir / "pca_structure.json", "w") as f:
        json.dump(pca_results, f, indent=2)
    logger.info(f"PCA structure saved to {results_dir / 'pca_structure.json'}")

    # Also save PCA projections for later analysis
    np.savez(
        results_dir / "pca_projections.npz",
        projections=projections,
        emotions=np.array(sorted(primary_vectors.keys())),
    )

    # Plots
    plot_cosine_similarity_matrix(sim_matrix, emo_list, results_dir / "cosine_similarity.png")
    plot_pca_scatter(projections, sorted(primary_vectors.keys()), results_dir / "pca_scatter.png")

    # ----------------------------------------------------------------
    # Step 7: KEY CHECK 2 — Cross-validation (all layers)
    # ----------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("KEY CHECK 2: CROSS-VALIDATION ACCURACY")
    logger.info(f"{'='*60}")

    cv_results = {}
    for layer_idx in all_layers:
        logger.info(f"\n  --- Layer {layer_idx} ---")
        cv = validate_cross_validation(
            model, stories, layer_idx,
            token_offset=settings.token_offset,
            test_fraction=0.2,
        )
        cv_results[str(layer_idx)] = {
            "accuracy": cv.value,
            "passed": cv.passed,
            "threshold": cv.threshold,
            "details": cv.details,
        }
        logger.info(f"  Layer {layer_idx}: accuracy={cv.value:.1%} "
                    f"(threshold {cv.threshold:.0%}) {'PASS' if cv.passed else 'FAIL'}")
        logger.info(f"  {cv.details}")

        # Save incrementally after each layer
        with open(results_dir / "cross_validation.json", "w") as f:
            json.dump(cv_results, f, indent=2)

    logger.info(f"\nCross-validation results saved to {results_dir / 'cross_validation.json'}")

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 1 SUMMARY")
    logger.info(f"{'='*60}")

    summary = {
        "model": model_config.name,
        "n_emotions": len(stories),
        "n_stories": total_stories,
        "n_neutral": len(neutral),
        "primary_layer": primary_layer,
        "all_layers": all_layers,
        "pca": {
            "pc1_valence_r": valence_result.value,
            "pc1_valence_passed": valence_result.passed,
            "pc2_arousal_r": arousal_result.value,
            "pc2_arousal_passed": arousal_result.passed,
        },
        "cross_validation": {
            str(k): v["accuracy"] for k, v in cv_results.items()
        },
        "validation_battery": {r["test"]: r["passed"] for r in val_summary},
    }

    all_passed = all(r.passed for r in validation_results)
    critical_passed = all(
        r.passed for r in validation_results
        if r.test_name in ["cosine_similarity_opposites", "logit_lens_match"]
    )
    summary["all_tests_passed"] = all_passed
    summary["critical_tests_passed"] = critical_passed

    with open(results_dir / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Full summary saved to {results_dir / 'phase1_summary.json'}")

    for key, val in summary["pca"].items():
        logger.info(f"  {key}: {val}")
    for layer, acc in summary["cross_validation"].items():
        logger.info(f"  CV layer {layer}: {acc:.1%}")

    if critical_passed:
        logger.info("\n✓ Phase 1 PASSED. Proceed to Phase 2.")
        logger.info(f"  python scripts/02_run_phase2.py --model {args.model}")
    else:
        logger.warning("\n✗ Phase 1 FAILED critical tests. Review results before proceeding.")

    model.cleanup()
    logger.info("Done.")


if __name__ == "__main__":
    main()
