"""
Generate emotional stories and neutral dialogues for emotion vector extraction.

Follows the Anthropic paper's methodology:
  - Uses the model itself to generate stories (self-probing approach)
  - Stories are seeded by (topic, emotion) pairs
  - Multiple stories per prompt, parsed apart afterward
  - Neutral dialogues generated separately for denoising
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from loguru import logger

from config import (
    STORY_GENERATION_PROMPT,
    NEUTRAL_DIALOGUE_PROMPT,
    STORIES_DIR,
    NEUTRAL_DIR,
    ensure_dirs,
)


def parse_stories(raw_text: str, expected_count: int) -> List[str]:
    """
    Parse multiple stories from a single generation.

    The prompt asks the model to format as:
      [story 1]
      [story 2]
      ...

    We split on the [story N] markers. If markers are absent,
    fall back to splitting on double newlines.
    """
    # Try to split on [story N] or [Story N] markers
    pattern = r'\[(?:[Ss]tory\s*\d+)\]'
    parts = re.split(pattern, raw_text)

    # First element is usually empty or preamble; drop it
    stories = [p.strip() for p in parts if p.strip()]

    if len(stories) >= expected_count:
        return stories[:expected_count]

    # Fallback: split on numbered markers like "1." or "1)"
    pattern2 = r'\n\s*\d+[\.\)]\s*\n'
    parts2 = re.split(pattern2, raw_text)
    stories2 = [p.strip() for p in parts2 if p.strip() and len(p.strip()) > 50]
    if len(stories2) >= expected_count:
        return stories2[:expected_count]

    # Last fallback: split on triple newlines
    parts3 = raw_text.split("\n\n\n")
    stories3 = [p.strip() for p in parts3 if p.strip() and len(p.strip()) > 50]
    if len(stories3) >= expected_count:
        return stories3[:expected_count]

    # If we still don't have enough, return what we have
    # (will be logged as a warning by the caller)
    all_candidates = stories or stories2 or stories3 or [raw_text.strip()]
    return all_candidates


def generate_stories(
    model,  # ModelWrapper
    emotions: List[str],
    topics: List[str],
    stories_per_topic: int = 12,
    output_dir: Optional[Path] = None,
    max_new_tokens: int = 3000,
) -> Dict[str, List[Dict]]:
    """
    Generate emotional stories for all (emotion, topic) pairs.

    Args:
        model: ModelWrapper instance (used for generation only).
        emotions: List of emotion words.
        topics: List of topic descriptions.
        stories_per_topic: How many stories per (emotion, topic) pair.
        output_dir: Where to save the JSON files.

    Returns:
        Dict mapping emotion -> list of {text, topic, emotion, story_idx} dicts.
    """
    if output_dir is None:
        output_dir = STORIES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stories = {}
    total = len(emotions) * len(topics)
    pbar = tqdm(total=total, desc="Generating stories")

    for emotion in emotions:
        emotion_stories = []
        emotion_file = output_dir / f"{emotion.replace(' ', '_')}.json"

        # Skip if already generated (all topics complete)
        if emotion_file.exists():
            with open(emotion_file) as f:
                emotion_stories = json.load(f)
            # Check if all topics are covered
            done_topics = set(s["topic"] for s in emotion_stories)
            remaining_topics = [t for t in topics if t not in done_topics]
            if not remaining_topics:
                logger.info(f"Loaded {len(emotion_stories)} existing stories for '{emotion}'")
                all_stories[emotion] = emotion_stories
                pbar.update(len(topics))
                continue
            else:
                logger.info(
                    f"Resuming '{emotion}': {len(done_topics)}/{len(topics)} topics done, "
                    f"{len(remaining_topics)} remaining"
                )
                pbar.update(len(done_topics))
        else:
            remaining_topics = list(topics)

        for topic in remaining_topics:
            prompt = STORY_GENERATION_PROMPT.format(
                n_stories=stories_per_topic,
                topic=topic,
                emotion=emotion,
            )

            try:
                raw = model.generate_plain(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=0.8,
                    top_p=0.95,
                    do_sample=True,
                )
            except Exception as e:
                logger.error(f"Generation failed for ({emotion}, {topic}): {e}")
                pbar.update(1)
                continue

            parsed = parse_stories(raw, stories_per_topic)

            if len(parsed) < stories_per_topic:
                logger.warning(
                    f"Only parsed {len(parsed)}/{stories_per_topic} stories "
                    f"for ({emotion}, {topic})"
                )

            for i, story_text in enumerate(parsed):
                emotion_stories.append({
                    "text": story_text,
                    "topic": topic,
                    "emotion": emotion,
                    "story_idx": i,
                })

            # Save after each topic so nothing is lost
            with open(emotion_file, "w") as f:
                json.dump(emotion_stories, f, indent=2)

            pbar.update(1)

        logger.info(f"Saved {len(emotion_stories)} stories for '{emotion}'")
        all_stories[emotion] = emotion_stories

    pbar.close()
    return all_stories


def generate_neutral_dialogues(
    model,
    topics: List[str],
    dialogues_per_topic: int = 5,
    output_dir: Optional[Path] = None,
    max_new_tokens: int = 2000,
) -> List[Dict]:
    """
    Generate emotionally neutral dialogues for PCA denoising.

    These are used to compute principal components of activations that capture
    non-emotional variance. These PCs are projected out of the emotion vectors.
    """
    if output_dir is None:
        output_dir = NEUTRAL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "neutral_dialogues.json"
    if output_file.exists():
        with open(output_file) as f:
            dialogues = json.load(f)
        logger.info(f"Loaded {len(dialogues)} existing neutral dialogues")
        return dialogues

    dialogues = []
    for topic in tqdm(topics, desc="Generating neutral dialogues"):
        prompt = NEUTRAL_DIALOGUE_PROMPT.format(
            n_stories=dialogues_per_topic,
            topic=topic,
        )

        try:
            raw = model.generate_plain(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
            )
        except Exception as e:
            logger.error(f"Neutral generation failed for {topic}: {e}")
            continue

        # Convert Person/AI to Human/Assistant (as the paper does post-hoc)
        raw = raw.replace("Person:", "Human:").replace("AI:", "Assistant:")

        # Split into individual dialogues
        parts = re.split(r'\n\s*\n\s*\n', raw)
        for i, part in enumerate(parts):
            part = part.strip()
            if part and len(part) > 30:
                dialogues.append({
                    "text": part,
                    "topic": topic,
                    "dialogue_idx": i,
                })

        # Save incrementally after each topic
        with open(output_file, "w") as f:
            json.dump(dialogues, f, indent=2)

    logger.info(f"Saved {len(dialogues)} neutral dialogues")
    return dialogues
