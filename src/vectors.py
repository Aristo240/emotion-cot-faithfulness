"""
Emotion vector extraction from model activations.

Methodology (following the Anthropic paper):
  1. For each (emotion, topic) pair, extract residual stream activations
     averaged across token positions from the 50th token onward.
  2. Compute per-emotion mean activation across all stories for that emotion.
  3. Subtract the grand mean across all emotions.
  4. Compute top PCs of activations on neutral transcripts (enough to
     explain 50% of variance). Project these out of the emotion vectors.
  5. The resulting directions are the "emotion vectors" / "emotion probes."
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from loguru import logger

from config import ACTIVATIONS_DIR, VECTORS_DIR, ensure_dirs


def extract_story_activations(
    model,  # ModelWrapper
    stories: Dict[str, List[Dict]],
    layer_indices: List[int],
    token_offset: int = 50,
    output_dir: Optional[Path] = None,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Extract mean activations for each emotion's stories at specified layers.

    For memory efficiency, we accumulate running means rather than storing
    all per-story activations.

    Returns:
        Dict mapping emotion -> {layer_idx -> mean_activation (hidden_dim,)}
    """
    if output_dir is None:
        output_dir = ACTIVATIONS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    emotion_means = {}

    for emotion, story_list in tqdm(stories.items(), desc="Extracting activations"):
        cache_file = output_dir / f"mean_acts_{emotion.replace(' ', '_')}.npz"
        if cache_file.exists():
            loaded = np.load(cache_file)
            emotion_means[emotion] = {
                int(k.split("_")[1]): loaded[k] for k in loaded.files
            }
            logger.info(f"Loaded cached activations for '{emotion}'")
            continue

        # Running mean computation
        running_sum = {idx: None for idx in layer_indices}
        count = 0

        for story in story_list:
            text = story["text"]
            if not text or len(text) < 20:
                continue

            try:
                acts = model.extract_mean_activations(
                    text, layer_indices, token_offset
                )
            except Exception as e:
                logger.warning(f"Extraction failed for story ({emotion}): {e}")
                continue

            for idx in layer_indices:
                v = acts[idx].numpy().astype(np.float32)
                if running_sum[idx] is None:
                    running_sum[idx] = v.copy()
                else:
                    running_sum[idx] += v
            count += 1

        if count == 0:
            logger.error(f"No valid stories for '{emotion}'!")
            continue

        means = {}
        save_dict = {}
        for idx in layer_indices:
            m = running_sum[idx] / count
            means[idx] = m
            save_dict[f"layer_{idx}"] = m

        np.savez(cache_file, **save_dict)
        emotion_means[emotion] = means
        logger.info(f"Extracted activations for '{emotion}' from {count} stories")

    return emotion_means


def extract_neutral_activations(
    model,
    dialogues: List[Dict],
    layer_indices: List[int],
    token_offset: int = 50,
    output_dir: Optional[Path] = None,
) -> Dict[int, np.ndarray]:
    """
    Extract activations from neutral dialogues for PCA denoising.

    Returns:
        Dict mapping layer_idx -> activations array (n_dialogues, hidden_dim).
    """
    if output_dir is None:
        output_dir = ACTIVATIONS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_file = output_dir / "neutral_activations.npz"
    if cache_file.exists():
        loaded = np.load(cache_file)
        result = {int(k.split("_")[1]): loaded[k] for k in loaded.files}
        logger.info(f"Loaded cached neutral activations ({loaded[loaded.files[0]].shape[0]} dialogues)")
        return result

    all_acts = {idx: [] for idx in layer_indices}

    for dlg in tqdm(dialogues, desc="Extracting neutral activations"):
        text = dlg["text"]
        if not text or len(text) < 20:
            continue
        try:
            acts = model.extract_mean_activations(text, layer_indices, token_offset)
        except Exception as e:
            logger.warning(f"Neutral extraction failed: {e}")
            continue

        for idx in layer_indices:
            all_acts[idx].append(acts[idx].numpy().astype(np.float32))

    result = {}
    save_dict = {}
    for idx in layer_indices:
        arr = np.stack(all_acts[idx], axis=0)
        result[idx] = arr
        save_dict[f"layer_{idx}"] = arr

    np.savez(cache_file, **save_dict)
    logger.info(f"Extracted neutral activations: {arr.shape}")
    return result


def compute_emotion_vectors(
    emotion_means: Dict[str, Dict[int, np.ndarray]],
    neutral_activations: Dict[int, np.ndarray],
    layer_indices: List[int],
    variance_threshold: float = 0.50,
    output_dir: Optional[Path] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Compute emotion vectors following the paper's procedure:
      1. Per-emotion mean - grand mean = raw emotion vector
      2. Compute PCA of neutral activations
      3. Project out top PCs (enough to explain `variance_threshold` of variance)

    Returns:
        Dict mapping layer_idx -> {emotion -> vector (hidden_dim,)}
    """
    if output_dir is None:
        output_dir = VECTORS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    emotions = list(emotion_means.keys())
    vectors = {}

    for layer_idx in tqdm(layer_indices, desc="Computing emotion vectors"):
        cache_file = output_dir / f"emotion_vectors_layer_{layer_idx}.npz"
        if cache_file.exists():
            loaded = np.load(cache_file)
            vectors[layer_idx] = {k: loaded[k] for k in loaded.files}
            logger.info(f"Loaded cached vectors for layer {layer_idx}")
            continue

        # Step 1: Compute grand mean
        all_means = []
        for emo in emotions:
            if layer_idx in emotion_means[emo]:
                all_means.append(emotion_means[emo][layer_idx])
        all_means = np.stack(all_means, axis=0)  # (n_emotions, hidden_dim)
        grand_mean = all_means.mean(axis=0)  # (hidden_dim,)

        # Step 2: Raw emotion vectors (mean - grand mean)
        raw_vectors = {}
        for emo in emotions:
            if layer_idx in emotion_means[emo]:
                raw_vectors[emo] = emotion_means[emo][layer_idx] - grand_mean

        # Step 3: PCA of neutral activations
        neutral = neutral_activations[layer_idx]  # (n_dialogues, hidden_dim)
        neutral_centered = neutral - neutral.mean(axis=0, keepdims=True)

        # SVD for PCA (more numerically stable than covariance)
        U, S, Vt = np.linalg.svd(neutral_centered, full_matrices=False)
        explained_var = (S ** 2) / (S ** 2).sum()
        cumulative_var = np.cumsum(explained_var)
        n_components = np.searchsorted(cumulative_var, variance_threshold) + 1
        n_components = min(n_components, len(S))

        logger.info(
            f"Layer {layer_idx}: projecting out {n_components} neutral PCs "
            f"(explaining {cumulative_var[n_components-1]:.1%} variance)"
        )

        # Step 4: Project out neutral PCs from emotion vectors
        projection_basis = Vt[:n_components].T  # (hidden_dim, n_components)
        layer_vectors = {}
        for emo, raw_v in raw_vectors.items():
            # Project out: v_clean = v - P @ P^T @ v
            # where P = projection_basis
            projected = projection_basis @ (projection_basis.T @ raw_v)
            clean_v = raw_v - projected
            layer_vectors[emo] = clean_v.astype(np.float32)

        np.savez(cache_file, **layer_vectors)
        vectors[layer_idx] = layer_vectors
        logger.info(f"Computed {len(layer_vectors)} emotion vectors for layer {layer_idx}")

    return vectors


def load_emotion_vectors(
    layer_idx: int,
    vectors_dir: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Load precomputed emotion vectors for a single layer."""
    if vectors_dir is None:
        vectors_dir = VECTORS_DIR
    cache_file = vectors_dir / f"emotion_vectors_layer_{layer_idx}.npz"
    if not cache_file.exists():
        raise FileNotFoundError(f"No vectors found at {cache_file}")
    loaded = np.load(cache_file)
    return {k: loaded[k] for k in loaded.files}


def compute_probe_projection(
    activation: np.ndarray,
    emotion_vector: np.ndarray,
) -> float:
    """
    Compute the projection of an activation onto an emotion vector.
    This is the "emotion probe" value used throughout the paper.

    Uses cosine similarity (not raw dot product) for comparability
    across layers and emotions.
    """
    a_norm = np.linalg.norm(activation)
    v_norm = np.linalg.norm(emotion_vector)
    if a_norm < 1e-8 or v_norm < 1e-8:
        return 0.0
    return float(np.dot(activation, emotion_vector) / (a_norm * v_norm))
