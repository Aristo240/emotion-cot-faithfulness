#!/usr/bin/env python3
"""
PILOT: Run this FIRST. Takes ~2-4 hours on 8 GPUs.

Tests whether emotion vectors exist in your chosen model before
committing to the full study.

Steps:
  1. Generate 100 stories (5 emotions × 5 topics × 4 stories)
  2. Generate 25 neutral dialogues (5 topics × 5 dialogues)
  3. Extract activations at 3 layers
  4. Compute emotion vectors
  5. Check cosine similarity structure
  6. Logit lens check
  7. Print GO / NO-GO decision

Usage:
    python scripts/00_pilot.py --model llama-70b
    python scripts/00_pilot.py --model qwen-72b
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODELS, PILOT_EMOTIONS, PILOT_TOPICS, ensure_dirs,
    DATA_DIR, RESULTS_DIR,
)
from src.model import ModelWrapper
from src.generate import generate_stories, generate_neutral_dialogues
from src.vectors import (
    extract_story_activations,
    extract_neutral_activations,
    compute_emotion_vectors,
)
from src.validate import (
    validate_cosine_similarity,
    validate_pca_vs_human,
    validate_logit_lens,
    validate_clustering,
)
from src.analysis import plot_cosine_similarity_matrix, plot_pca_scatter

from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Pilot: check for emotion vector signal")
    parser.add_argument(
        "--model", type=str, required=True, choices=list(MODELS.keys()),
        help="Model to test (e.g., 'llama-70b' or 'qwen-72b')"
    )
    parser.add_argument(
        "--stories-per-topic", type=int, default=4,
        help="Stories per (emotion, topic) pair (default: 4)"
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="HuggingFace cache directory for model weights"
    )
    args = parser.parse_args()

    ensure_dirs()
    model_config = MODELS[args.model]

    # Create pilot-specific output dirs
    pilot_dir = DATA_DIR / "pilot" / model_config.short_name
    pilot_dir.mkdir(parents=True, exist_ok=True)
    results_dir = RESULTS_DIR / "pilot" / model_config.short_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Step 1: Load model
    # ----------------------------------------------------------------
    logger.info(f"Loading {model_config.name}...")
    model = ModelWrapper(model_config.name, cache_dir=args.cache_dir)
    logger.info(f"Model has {model.num_layers} layers, hidden_dim={model.hidden_dim}")

    # Verify layer indices
    pilot_layers = model_config.pilot_layers
    logger.info(f"Pilot layers: {pilot_layers}")

    # ----------------------------------------------------------------
    # Step 2: Generate stories
    # ----------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info(f"Generating stories: {len(PILOT_EMOTIONS)} emotions × {len(PILOT_TOPICS)} topics × {args.stories_per_topic} stories")
    logger.info(f"{'='*60}")

    stories = generate_stories(
        model,
        emotions=PILOT_EMOTIONS,
        topics=PILOT_TOPICS,
        stories_per_topic=args.stories_per_topic,
        output_dir=pilot_dir / "stories",
    )

    total_stories = sum(len(s) for s in stories.values())
    logger.info(f"Total stories generated: {total_stories}")

    # ----------------------------------------------------------------
    # Step 3: Generate neutral dialogues
    # ----------------------------------------------------------------
    logger.info(f"\nGenerating {len(PILOT_TOPICS) * 5} neutral dialogues...")
    neutral = generate_neutral_dialogues(
        model,
        topics=PILOT_TOPICS,
        dialogues_per_topic=5,
        output_dir=pilot_dir / "neutral",
    )
    logger.info(f"Total neutral dialogues: {len(neutral)}")

    # ----------------------------------------------------------------
    # Step 4: Extract activations
    # ----------------------------------------------------------------
    logger.info(f"\nExtracting activations at layers {pilot_layers}...")
    emotion_means = extract_story_activations(
        model, stories, pilot_layers, token_offset=50,
        output_dir=pilot_dir / "activations",
    )
    neutral_acts = extract_neutral_activations(
        model, neutral, pilot_layers, token_offset=50,
        output_dir=pilot_dir / "activations",
    )

    # ----------------------------------------------------------------
    # Step 5: Compute emotion vectors
    # ----------------------------------------------------------------
    logger.info("\nComputing emotion vectors...")
    vectors = compute_emotion_vectors(
        emotion_means, neutral_acts, pilot_layers,
        variance_threshold=0.50,
        output_dir=pilot_dir / "vectors",
    )

    # ----------------------------------------------------------------
    # Step 6: Quick validation checks
    # ----------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("PILOT VALIDATION")
    logger.info(f"{'='*60}")

    # Use the mid-late layer (primary analysis layer from the paper)
    mid_late = model_config.mid_late_layer
    # If mid_late not in pilot layers, use closest
    closest_layer = min(pilot_layers, key=lambda x: abs(x - mid_late))
    logger.info(f"Using layer {closest_layer} for validation (closest to mid-late={mid_late})")

    layer_vectors = vectors[closest_layer]
    logger.info(f"Emotion vectors computed for: {list(layer_vectors.keys())}")

    # Check 1: Cosine similarities between pilot emotions
    logger.info("\n--- Cosine Similarity Check ---")
    sim_matrix, emotion_list, cos_result = validate_cosine_similarity(layer_vectors)

    # Print the matrix
    logger.info("Cosine similarity matrix:")
    header = "         " + " ".join(f"{e[:6]:>7s}" for e in emotion_list)
    logger.info(header)
    for i, e1 in enumerate(emotion_list):
        row = f"{e1[:8]:<9s}" + " ".join(f"{sim_matrix[i,j]:>7.3f}" for j in range(len(emotion_list)))
        logger.info(row)

    # Key checks for the 5 pilot emotions:
    checks = []
    if "happy" in emotion_list and "sad" in emotion_list:
        i, j = emotion_list.index("happy"), emotion_list.index("sad")
        val = sim_matrix[i, j]
        passed = val < 0
        checks.append(f"happy-sad: {val:.3f} ({'PASS' if passed else 'FAIL'}: should be negative)")

    if "calm" in emotion_list and "desperate" in emotion_list:
        i, j = emotion_list.index("calm"), emotion_list.index("desperate")
        val = sim_matrix[i, j]
        passed = val < 0
        checks.append(f"calm-desperate: {val:.3f} ({'PASS' if passed else 'FAIL'}: should be negative)")

    if "calm" in emotion_list and "angry" in emotion_list:
        i, j = emotion_list.index("calm"), emotion_list.index("angry")
        val = sim_matrix[i, j]
        passed = val < 0
        checks.append(f"calm-angry: {val:.3f} ({'PASS' if passed else 'FAIL'}: should be negative)")

    for c in checks:
        logger.info(f"  {c}")

    # Check 2: Logit lens
    logger.info("\n--- Logit Lens Check ---")
    logit_result = validate_logit_lens(model, layer_vectors)
    logger.info(f"  {logit_result.details}")

    # Check 3: Check across ALL pilot layers
    logger.info("\n--- Cross-Layer Consistency ---")
    for layer_idx in pilot_layers:
        lv = vectors[layer_idx]
        if "happy" in lv and "sad" in lv:
            cos = np.dot(lv["happy"], lv["sad"]) / (
                np.linalg.norm(lv["happy"]) * np.linalg.norm(lv["sad"]) + 1e-8
            )
            logger.info(f"  Layer {layer_idx}: happy-sad cosine = {cos:.3f}")

    # ----------------------------------------------------------------
    # Step 7: Save plots
    # ----------------------------------------------------------------
    plot_cosine_similarity_matrix(sim_matrix, emotion_list, results_dir / "pilot_cosine_sim.png")

    # ----------------------------------------------------------------
    # GO / NO-GO decision
    # ----------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("PILOT DECISION")
    logger.info(f"{'='*60}")

    # Critical: happy-sad should be negative, calm-desperate should be negative
    go = True
    if "happy" in emotion_list and "sad" in emotion_list:
        i, j = emotion_list.index("happy"), emotion_list.index("sad")
        if sim_matrix[i, j] >= 0:
            go = False
            logger.error("FAIL: happy-sad similarity is not negative")
    if "calm" in emotion_list and "desperate" in emotion_list:
        i, j = emotion_list.index("calm"), emotion_list.index("desperate")
        if sim_matrix[i, j] >= 0:
            go = False
            logger.error("FAIL: calm-desperate similarity is not negative")

    if not logit_result.passed:
        logger.warning(f"WARNING: Logit lens match rate low ({logit_result.value:.1%})")
        # Not a hard fail for pilot with only 5 emotions

    if go:
        logger.info("✓ GO: Basic emotion vector structure detected. Proceed to full Phase 1.")
        logger.info(f"  Recommended: python scripts/01_run_phase1.py --model {args.model}")
    else:
        logger.warning("✗ NO-GO: Emotion vectors do not show expected structure.")
        logger.warning(f"  Try: python scripts/00_pilot.py --model {'qwen-72b' if 'llama' in args.model else 'llama-70b'}")

    # Save summary
    summary = {
        "model": model_config.name,
        "pilot_layers": pilot_layers,
        "n_emotions": len(PILOT_EMOTIONS),
        "n_topics": len(PILOT_TOPICS),
        "n_stories": total_stories,
        "n_neutral": len(neutral),
        "cosine_checks": checks,
        "logit_lens_match_rate": logit_result.value,
        "go_decision": go,
    }
    with open(results_dir / "pilot_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    model.cleanup()
    logger.info("Done.")


if __name__ == "__main__":
    main()
