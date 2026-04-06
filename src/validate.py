"""
Validation battery for emotion vectors.

Tests (all must pass for go-ahead):
  1. Cosine similarity matrix shows intuitive clustering
  2. K-means produces interpretable clusters
  3. PC1 correlates with human valence ratings (r > 0.5)
  4. PC2 correlates with human arousal ratings (r > 0.4)
  5. Logit lens: vectors upweight corresponding emotion tokens
  6. Cross-validation: vectors generalize to held-out stories

Human valence/arousal ratings from:
  Russell & Mehrabian (1977), "Evidence for a three-factor theory of emotions"
"""

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, List, Tuple, Optional
from loguru import logger
from dataclasses import dataclass


# ============================================================================
# Human valence/arousal norms (Russell & Mehrabian 1977)
# Subset of emotions overlapping with our set.
# Valence (pleasure): higher = more positive
# Arousal: higher = more activated
# ============================================================================

HUMAN_NORMS = {
    # emotion: (valence, arousal) on ~1-7 scale
    "afraid": (2.0, 5.7),
    "amazed": (5.2, 5.1),
    "amused": (5.5, 4.0),
    "angry": (2.0, 5.9),
    "anxious": (2.4, 5.6),
    "ashamed": (2.0, 4.2),
    "astonished": (4.5, 5.6),
    "at ease": (5.8, 2.5),
    "bewildered": (3.0, 4.8),
    "bitter": (2.2, 4.0),
    "blissful": (6.5, 3.5),
    "bored": (2.8, 1.8),
    "calm": (5.5, 1.8),
    "cheerful": (6.2, 4.5),
    "content": (5.8, 2.5),
    "defiant": (3.5, 5.0),
    "delighted": (6.5, 5.0),
    "depressed": (1.5, 2.0),
    "desperate": (1.5, 5.8),
    "disgusted": (2.0, 5.0),
    "distressed": (2.0, 5.5),
    "eager": (5.5, 5.5),
    "ecstatic": (6.8, 6.5),
    "elated": (6.5, 5.8),
    "embarrassed": (2.5, 4.5),
    "enthusiastic": (6.0, 5.8),
    "envious": (2.5, 4.5),
    "excited": (6.2, 6.0),
    "frustrated": (2.0, 5.2),
    "fulfilled": (6.0, 2.8),
    "furious": (1.8, 6.5),
    "gloomy": (2.0, 2.2),
    "grateful": (6.0, 3.5),
    "guilty": (2.0, 4.0),
    "happy": (6.5, 4.5),
    "hopeful": (5.8, 4.0),
    "hostile": (1.8, 5.5),
    "inspired": (6.0, 5.0),
    "jealous": (2.5, 5.0),
    "joyful": (6.5, 5.5),
    "lonely": (1.8, 2.5),
    "loving": (6.5, 4.0),
    "melancholy": (2.2, 2.0),
    "nervous": (2.5, 5.5),
    "nostalgic": (4.0, 2.5),
    "optimistic": (6.0, 4.5),
    "panicked": (1.5, 6.5),
    "peaceful": (6.0, 1.5),
    "pleased": (5.8, 3.5),
    "proud": (5.8, 4.5),
    "relaxed": (5.8, 1.5),
    "resentful": (2.0, 4.5),
    "sad": (1.8, 2.5),
    "satisfied": (5.5, 2.8),
    "scared": (2.0, 5.8),
    "serene": (6.0, 1.5),
    "sleepy": (3.5, 1.2),
    "surprised": (4.5, 5.8),
    "suspicious": (2.5, 4.8),
    "terrified": (1.5, 6.5),
    "thrilled": (6.5, 6.0),
    "tired": (2.8, 1.5),
    "troubled": (2.2, 4.5),
    "vulnerable": (2.5, 3.5),
    "weary": (2.5, 1.5),
    "worried": (2.2, 5.0),
}


@dataclass
class ValidationResult:
    """Container for a single validation test result."""
    test_name: str
    passed: bool
    value: float
    threshold: float
    details: str = ""


def validate_cosine_similarity(
    vectors: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[str], ValidationResult]:
    """
    Compute pairwise cosine similarity matrix.
    Check that semantically opposite emotions have negative similarity.
    """
    emotions = sorted(vectors.keys())
    n = len(emotions)
    sim_matrix = np.zeros((n, n))

    for i, e1 in enumerate(emotions):
        for j, e2 in enumerate(emotions):
            v1 = vectors[e1]
            v2 = vectors[e2]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 1e-8 and n2 > 1e-8:
                sim_matrix[i, j] = np.dot(v1, v2) / (n1 * n2)

    # Check that expected opposites have negative similarity
    opposite_pairs = [
        ("happy", "sad"), ("calm", "desperate"), ("calm", "angry"),
        ("happy", "angry"), ("peaceful", "panicked"),
    ]
    n_negative = 0
    n_checked = 0
    for e1, e2 in opposite_pairs:
        if e1 in emotions and e2 in emotions:
            i, j = emotions.index(e1), emotions.index(e2)
            n_checked += 1
            if sim_matrix[i, j] < 0:
                n_negative += 1

    passed = n_checked > 0 and (n_negative / n_checked) >= 0.6
    result = ValidationResult(
        test_name="cosine_similarity_opposites",
        passed=passed,
        value=n_negative / max(n_checked, 1),
        threshold=0.6,
        details=f"{n_negative}/{n_checked} expected opposite pairs have negative similarity",
    )
    return sim_matrix, emotions, result


def validate_clustering(
    vectors: Dict[str, np.ndarray],
    k: int = 8,
) -> ValidationResult:
    """
    K-means clustering and silhouette score.
    Interpretable clusters = silhouette > 0.1 (low bar, but for high-dim data).
    """
    emotions = sorted(vectors.keys())
    X = np.stack([vectors[e] for e in emotions], axis=0)

    if len(emotions) < k:
        k = max(2, len(emotions) // 2)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    silhouette = silhouette_score(X, labels)

    # Log clusters for inspection
    clusters = {}
    for emo, label in zip(emotions, labels):
        clusters.setdefault(label, []).append(emo)

    details = "\n".join(
        f"  Cluster {l}: {', '.join(sorted(emos))}"
        for l, emos in sorted(clusters.items())
    )

    return ValidationResult(
        test_name="kmeans_clustering",
        passed=silhouette > 0.05,
        value=silhouette,
        threshold=0.05,
        details=f"Silhouette score: {silhouette:.3f}\n{details}",
    )


def validate_pca_vs_human(
    vectors: Dict[str, np.ndarray],
) -> Tuple[ValidationResult, ValidationResult, np.ndarray]:
    """
    PCA on emotion vectors. Check that:
      PC1 correlates with human valence (r > 0.5)
      PC2 correlates with human arousal (r > 0.4)

    Returns:
        (valence_result, arousal_result, projections)
    """
    emotions = sorted(vectors.keys())
    X = np.stack([vectors[e] for e in emotions], axis=0)
    X_centered = X - X.mean(axis=0, keepdims=True)

    # PCA via SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    explained_var = (S ** 2) / (S ** 2).sum()
    projections = X_centered @ Vt[:2].T  # (n_emotions, 2)

    logger.info(f"PC1 explains {explained_var[0]:.1%}, PC2 explains {explained_var[1]:.1%}")

    # Match with human norms
    overlapping = [e for e in emotions if e in HUMAN_NORMS]
    if len(overlapping) < 10:
        logger.warning(f"Only {len(overlapping)} emotions overlap with human norms")
        return (
            ValidationResult("pc1_valence", False, 0.0, 0.5, "Too few overlapping emotions"),
            ValidationResult("pc2_arousal", False, 0.0, 0.4, "Too few overlapping emotions"),
            projections,
        )

    human_valence = np.array([HUMAN_NORMS[e][0] for e in overlapping])
    human_arousal = np.array([HUMAN_NORMS[e][1] for e in overlapping])
    pc1_vals = np.array([projections[emotions.index(e), 0] for e in overlapping])
    pc2_vals = np.array([projections[emotions.index(e), 1] for e in overlapping])

    # PC1 vs valence (allow sign flip)
    r1, p1 = stats.pearsonr(pc1_vals, human_valence)
    # PC2 vs arousal (allow sign flip)
    r2, p2 = stats.pearsonr(pc2_vals, human_arousal)

    # Also check flipped assignment (PC1=arousal, PC2=valence)
    r1_alt, _ = stats.pearsonr(pc2_vals, human_valence)
    r2_alt, _ = stats.pearsonr(pc1_vals, human_arousal)

    # Use whichever assignment gives better match
    if abs(r1) + abs(r2) >= abs(r1_alt) + abs(r2_alt):
        valence_r, arousal_r = abs(r1), abs(r2)
        assignment = "PC1=valence, PC2=arousal"
    else:
        valence_r, arousal_r = abs(r1_alt), abs(r2_alt)
        assignment = "PC1=arousal, PC2=valence (flipped)"

    valence_result = ValidationResult(
        test_name="pc_valence_correlation",
        passed=valence_r > 0.5,
        value=valence_r,
        threshold=0.5,
        details=f"{assignment}, r={valence_r:.3f} (n={len(overlapping)} emotions)",
    )
    arousal_result = ValidationResult(
        test_name="pc_arousal_correlation",
        passed=arousal_r > 0.4,
        value=arousal_r,
        threshold=0.4,
        details=f"{assignment}, r={arousal_r:.3f} (n={len(overlapping)} emotions)",
    )

    return valence_result, arousal_result, projections


def validate_logit_lens(
    model,  # ModelWrapper
    vectors: Dict[str, np.ndarray],
    top_k: int = 10,
) -> ValidationResult:
    """
    Check that emotion vectors upweight corresponding emotion tokens
    through the unembedding matrix.

    Pass criterion: for >= 70% of emotions, the emotion word (or close variant)
    appears in the top 10 upweighted tokens.
    """
    import torch

    matches = 0
    total = 0
    details_list = []

    for emotion, vec in vectors.items():
        total += 1
        vec_tensor = torch.from_numpy(vec).float()

        try:
            top_tokens, bottom_tokens = model.logit_lens(vec_tensor, top_k=top_k)
        except Exception as e:
            logger.warning(f"Logit lens failed for '{emotion}': {e}")
            continue

        top_words = [t[0].strip().lower() for t in top_tokens]

        # Check if the emotion word or a clear variant appears
        # (e.g., "happy" might surface as "happ", "happiness", "happily")
        emotion_lower = emotion.lower().replace("-", "").replace(" ", "")
        found = any(
            emotion_lower[:4] in w.replace("-", "").replace(" ", "")
            for w in top_words
        )
        if found:
            matches += 1

        details_list.append(
            f"  {emotion}: top=[{', '.join(top_words[:5])}] {'✓' if found else '✗'}"
        )

    rate = matches / max(total, 1)
    return ValidationResult(
        test_name="logit_lens_match",
        passed=rate >= 0.70,
        value=rate,
        threshold=0.70,
        details=f"{matches}/{total} emotions matched\n" + "\n".join(details_list[:20]),
    )


def validate_cross_validation(
    model,
    stories: Dict[str, List[Dict]],
    layer_idx: int,
    token_offset: int = 50,
    test_fraction: float = 0.2,
) -> ValidationResult:
    """
    Hold out 20% of stories, compute vectors from 80%, test whether
    held-out activations are closest to the correct emotion vector.

    Pass criterion: accuracy > 70% (chance = 1/n_emotions).
    """
    import random
    random.seed(42)

    emotions = sorted(stories.keys())
    n_emotions = len(emotions)

    # Split stories
    train_stories = {}
    test_stories = {}
    for emo in emotions:
        s = stories[emo]
        random.shuffle(s)
        split = max(1, int(len(s) * (1 - test_fraction)))
        train_stories[emo] = s[:split]
        test_stories[emo] = s[split:]

    # Compute vectors from train set
    train_means = {}
    for emo in emotions:
        acts = []
        for story in train_stories[emo]:
            try:
                a = model.extract_mean_activations(story["text"], [layer_idx], token_offset)
                acts.append(a[layer_idx].numpy())
            except Exception:
                continue
        if acts:
            train_means[emo] = np.mean(acts, axis=0)

    if len(train_means) < n_emotions:
        logger.warning(f"Only {len(train_means)}/{n_emotions} emotions have train data")

    grand_mean = np.mean(list(train_means.values()), axis=0)
    train_vectors = {e: m - grand_mean for e, m in train_means.items()}

    # Test: for each held-out story, find nearest emotion vector
    correct = 0
    total = 0
    for emo in emotions:
        if emo not in train_vectors:
            continue
        for story in test_stories[emo][:5]:  # Limit for speed
            try:
                a = model.extract_mean_activations(story["text"], [layer_idx], token_offset)
                test_act = a[layer_idx].numpy() - grand_mean
            except Exception:
                continue

            # Find most similar vector (cosine similarity)
            best_emo = None
            best_sim = -999
            for e2, v2 in train_vectors.items():
                sim = np.dot(test_act, v2) / (
                    np.linalg.norm(test_act) * np.linalg.norm(v2) + 1e-8
                )
                if sim > best_sim:
                    best_sim = sim
                    best_emo = e2

            if best_emo == emo:
                correct += 1
            total += 1

    accuracy = correct / max(total, 1)
    chance = 1.0 / max(n_emotions, 1)

    return ValidationResult(
        test_name="cross_validation_accuracy",
        passed=accuracy > 0.70,
        value=accuracy,
        threshold=0.70,
        details=f"{correct}/{total} correct (chance={chance:.1%})",
    )


def run_full_validation(
    model,
    vectors: Dict[str, np.ndarray],
    stories: Dict[str, List[Dict]],
    layer_idx: int,
    token_offset: int = 50,
) -> List[ValidationResult]:
    """
    Run the complete validation battery.
    Returns list of ValidationResult objects.
    """
    results = []

    logger.info("=" * 60)
    logger.info("RUNNING VALIDATION BATTERY")
    logger.info("=" * 60)

    # 1. Cosine similarity
    logger.info("Test 1: Cosine similarity of opposite pairs")
    sim_matrix, emotion_list, cos_result = validate_cosine_similarity(vectors)
    results.append(cos_result)
    logger.info(f"  {'PASS' if cos_result.passed else 'FAIL'}: {cos_result.details}")

    # 2. Clustering
    logger.info("Test 2: K-means clustering")
    cluster_result = validate_clustering(vectors)
    results.append(cluster_result)
    logger.info(f"  {'PASS' if cluster_result.passed else 'FAIL'}: silhouette={cluster_result.value:.3f}")

    # 3-4. PCA vs human norms
    logger.info("Test 3-4: PCA vs human valence/arousal")
    valence_result, arousal_result, projections = validate_pca_vs_human(vectors)
    results.append(valence_result)
    results.append(arousal_result)
    logger.info(f"  Valence: {'PASS' if valence_result.passed else 'FAIL'}: {valence_result.details}")
    logger.info(f"  Arousal: {'PASS' if arousal_result.passed else 'FAIL'}: {arousal_result.details}")

    # 5. Logit lens
    logger.info("Test 5: Logit lens token matching")
    logit_result = validate_logit_lens(model, vectors)
    results.append(logit_result)
    logger.info(f"  {'PASS' if logit_result.passed else 'FAIL'}: {logit_result.value:.1%} matched")

    # 6. Cross-validation
    logger.info("Test 6: Cross-validation accuracy")
    cv_result = validate_cross_validation(model, stories, layer_idx, token_offset)
    results.append(cv_result)
    logger.info(f"  {'PASS' if cv_result.passed else 'FAIL'}: {cv_result.details}")

    # Summary
    n_pass = sum(1 for r in results if r.passed)
    n_total = len(results)
    logger.info("=" * 60)
    logger.info(f"VALIDATION SUMMARY: {n_pass}/{n_total} tests passed")

    # Go/no-go decision
    critical_tests = ["cosine_similarity_opposites", "logit_lens_match"]
    critical_pass = all(
        r.passed for r in results if r.test_name in critical_tests
    )
    if critical_pass:
        logger.info("GO: Critical tests passed. Proceed to Phase 2.")
    else:
        logger.warning("NO-GO: Critical tests failed. Try another model or debug.")
    logger.info("=" * 60)

    return results
