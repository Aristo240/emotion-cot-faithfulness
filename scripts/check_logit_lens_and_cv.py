#!/usr/bin/env python3
"""
Quick diagnostic: logit lens + cross-validation on pilot data.
Run after 00_pilot.py completes.

Usage:
    python scripts/check_logit_lens_and_cv.py --model llama-70b
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS, PILOT_EMOTIONS, DATA_DIR
from src.model import ModelWrapper
from src.vectors import load_emotion_vectors
from loguru import logger
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=20, help="Top-k tokens to inspect")
    args = parser.parse_args()

    model_config = MODELS[args.model]
    pilot_dir = DATA_DIR / "pilot" / model_config.short_name
    primary_layer = min(model_config.pilot_layers, key=lambda x: abs(x - model_config.mid_late_layer))

    # Load HuggingFace model (needed for logit lens and activation extraction)
    logger.info(f"Loading {model_config.name} for logit lens + cross-validation...")
    model = ModelWrapper(model_config.name, cache_dir=args.cache_dir)

    # Load emotion vectors
    vectors_dir = pilot_dir / "vectors"
    vectors_all_layers = {}
    for layer_idx in model_config.pilot_layers:
        fpath = vectors_dir / f"emotion_vectors_layer_{layer_idx}.npz"
        if fpath.exists():
            loaded = np.load(fpath)
            vectors_all_layers[layer_idx] = {k: loaded[k] for k in loaded.files}

    layer_vectors = vectors_all_layers[primary_layer]
    logger.info(f"Using layer {primary_layer}, emotions: {list(layer_vectors.keys())}")

    # ==================================================================
    # CHECK 1: Logit lens (detailed)
    # ==================================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"LOGIT LENS CHECK (top-{args.top_k} tokens per emotion)")
    logger.info(f"{'='*60}")

    for emotion in sorted(layer_vectors.keys()):
        vec = torch.from_numpy(layer_vectors[emotion]).float()
        top_tokens, bottom_tokens = model.logit_lens(vec, top_k=args.top_k)

        top_words = [f"{t[0].strip()}" for t in top_tokens]
        top_with_scores = [f"{t[0].strip()} ({t[1]:.1f})" for t in top_tokens[:10]]
        bottom_with_scores = [f"{t[0].strip()} ({t[1]:.1f})" for t in bottom_tokens[:5]]

        # Check match (same logic as validate.py but with more detail)
        emotion_lower = emotion.lower().replace("-", "").replace(" ", "")
        found_in = None
        for i, w in enumerate(top_words):
            if emotion_lower[:4] in w.lower().replace("-", "").replace(" ", ""):
                found_in = i
                break

        status = f"MATCH at rank {found_in}" if found_in is not None else "NO MATCH"
        logger.info(f"\n  [{emotion.upper()}] {status}")
        logger.info(f"    Top 10 upweighted: {', '.join(top_with_scores)}")
        logger.info(f"    Top 5 downweighted: {', '.join(bottom_with_scores)}")

    # Also check: do the top tokens make SEMANTIC sense even if the exact
    # word doesn't appear? (The paper notes this is common.)
    logger.info(f"\n  NOTE: Exact word match is a strict criterion. The paper notes that")
    logger.info(f"  emotion vectors often upweight semantically related tokens rather")
    logger.info(f"  than the exact emotion word. Check the top tokens above manually.")

    # Try across all pilot layers
    logger.info(f"\n{'='*60}")
    logger.info(f"LOGIT LENS ACROSS LAYERS")
    logger.info(f"{'='*60}")

    for layer_idx in model_config.pilot_layers:
        if layer_idx not in vectors_all_layers:
            continue
        lv = vectors_all_layers[layer_idx]
        matches = 0
        for emo, vec_np in lv.items():
            vec = torch.from_numpy(vec_np).float()
            top_tokens, _ = model.logit_lens(vec, top_k=args.top_k)
            top_words = [t[0].strip().lower() for t in top_tokens]
            emo_lower = emo.lower().replace("-", "").replace(" ", "")
            if any(emo_lower[:4] in w.replace("-", "").replace(" ", "") for w in top_words):
                matches += 1
        logger.info(f"  Layer {layer_idx}: {matches}/{len(lv)} matched (top-{args.top_k})")

    # ==================================================================
    # CHECK 2: Cross-validation
    # ==================================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"CROSS-VALIDATION CHECK")
    logger.info(f"{'='*60}")

    # Load stories
    stories = {}
    for emotion in PILOT_EMOTIONS:
        fpath = pilot_dir / "stories" / f"{emotion.replace(' ', '_')}.json"
        if fpath.exists():
            with open(fpath) as f:
                stories[emotion] = json.load(f)
            logger.info(f"  Loaded {len(stories[emotion])} stories for '{emotion}'")

    from src.validate import validate_cross_validation
    cv_result = validate_cross_validation(
        model, stories, primary_layer, token_offset=50, test_fraction=0.2
    )
    logger.info(f"\n  Cross-validation result: {'PASS' if cv_result.passed else 'FAIL'}")
    logger.info(f"  Accuracy: {cv_result.value:.1%} (threshold: {cv_result.threshold:.0%})")
    logger.info(f"  Details: {cv_result.details}")

    # Also try other layers
    for layer_idx in model_config.pilot_layers:
        if layer_idx == primary_layer:
            continue
        cv = validate_cross_validation(model, stories, layer_idx, token_offset=50)
        logger.info(f"  Layer {layer_idx}: accuracy={cv.value:.1%} {'PASS' if cv.passed else 'FAIL'}")

    model.cleanup()
    logger.info("\nDone.")


if __name__ == "__main__":
    main()
