#!/usr/bin/env python3
"""
Phase 2 — diverse Task A variants.

Addresses red-team finding F1: the original TASK_A_VARIANTS are 4
near-paraphrases of the same `fast_sum` timing-budget hack, which makes
LOGO cross-task generalization tests meaningless. This script runs the
same Phase 2 grid (emotion × strength × rollouts) on the mechanistically
distinct TASK_A_DIVERSE variants defined in config.py — five different
shortcut mechanisms (timing budget, hardcoded lookup, fake verifier,
silent spec drop, misleading-by-omission).

We deliberately DO NOT modify `phase2_steering.py` so the canonical
reproduction script for the original results stays untouched. Output
goes to a separate jsonl that the LLM judge (`run_judge_reclassification.py`)
and the H5 held-out script (`scripts/h5_holdout.py`) will consume after
re-running Phase 3.

Per-trial classification falls back to the LLM judge's `judge_classification`
field (already the source of truth for downstream analyses); the regex
`classification` here is a placeholder that just records "unclear" — the
hack patterns differ across mechanisms and writing dedicated regexes per
mechanism would just be a brittle re-implementation of the judge.

Usage (after Phase 4 blockers finish, GPUs free):
    nohup $PYTHON scripts/phase2_diverse.py > logs/phase2_diverse_<ts>.log 2>&1 &

Resumable via jsonl-append (same pattern as phase2_steering.py).
"""

import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODELS, TASK_A_DIVERSE, RESULTS_DIR, DATA_DIR,
)
from src.model import ModelWrapper, SteeringConfig
from scripts.phase2_steering import (
    STEERING_LAYER, STRENGTHS, EMOTIONS, ROLLOUTS_PER_CONDITION, TOKEN_OFFSET,
    format_chat_prompt, compute_emotion_probes, get_completed_keys, save_jsonl_append,
    load_jsonl, ts,
)


def classify_diverse_placeholder(response: str):
    """Placeholder — actual outcomes assigned by the LLM judge downstream."""
    return "unclear", []


def run_diverse(model, vecs, residual_norm, emotion_vectors, out_path):
    completed = get_completed_keys(out_path)
    if completed:
        print(f"[{ts()}] Resuming: {len(completed)} entries already complete")

    work = []
    for variant in TASK_A_DIVERSE:
        task_id = variant["id"]
        for emotion in EMOTIONS:
            for strength in STRENGTHS:
                for rollout in range(ROLLOUTS_PER_CONDITION):
                    if strength == 0.0 and emotion != EMOTIONS[0]:
                        continue
                    key = (
                        f"{task_id}_unsteered_r{rollout}" if strength == 0.0
                        else f"{task_id}_{emotion}_{strength:+.2f}_r{rollout}"
                    )
                    if key not in completed:
                        work.append({
                            "variant": variant, "task_id": task_id,
                            "emotion": emotion, "strength": strength,
                            "rollout": rollout, "key": key,
                        })

    print(f"[{ts()}] {len(work)} generations to run "
          f"({len(TASK_A_DIVERSE)} variants × grid)")

    for i, item in enumerate(work):
        variant = item["variant"]
        prompt = format_chat_prompt(
            model.tokenizer, variant["system"], variant["prompt"]
        )
        strength = item["strength"]
        emotion = item["emotion"]
        if strength == 0.0:
            steering_configs = None
        else:
            vec = torch.from_numpy(vecs[emotion].copy()).float()
            steering_configs = [SteeringConfig(
                vector=vec, layer_idx=STEERING_LAYER,
                strength=strength, residual_norm=residual_norm,
            )]
        t0 = time.time()
        error = None
        try:
            result = model.generate_steered(
                prompt=prompt, steering_configs=steering_configs,
                max_new_tokens=1024, temperature=0.7, top_p=0.95,
                do_sample=True, extract_layers=[STEERING_LAYER],
            )
            response = result["text"]
            classification, matches = classify_diverse_placeholder(response)
            act = result["activations"].get(STEERING_LAYER)
            emotion_probes = (compute_emotion_probes(act, emotion_vectors, TOKEN_OFFSET)
                              if act is not None else {})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            response, classification, matches, emotion_probes = "", "error", [], {}
            error = "OOM"
        except Exception as e:
            response, classification, matches, emotion_probes = "", "error", [], {}
            error = f"{type(e).__name__}: {str(e)[:300]}"

        elapsed = time.time() - t0
        entry = {
            "key": item["key"], "task_id": item["task_id"],
            "emotion": emotion if strength != 0.0 else "none",
            "strength": strength, "rollout": item["rollout"],
            "response": response, "classification": classification,
            "classification_matches": matches,
            "emotion_probes": emotion_probes,
            "elapsed_s": round(elapsed, 1), "timestamp": ts(),
        }
        if error:
            entry["error"] = error
        save_jsonl_append(out_path, entry)
        if (i + 1) % 10 == 0:
            print(f"[{ts()}] {i+1}/{len(work)} done")


def main():
    model_key = "llama-70b"
    cfg = MODELS[model_key]
    short = cfg.short_name

    out_dir = RESULTS_DIR / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "task_a_diverse.jsonl"

    vectors_path = DATA_DIR / "phase1" / short / "vectors" / f"emotion_vectors_layer_{STEERING_LAYER}.npz"
    norms_path = DATA_DIR / "phase1" / short / "vectors" / "residual_norms.npz"

    print(f"[{ts()}] Loading vectors and norms...")
    emotion_vectors = np.load(vectors_path)
    norms = np.load(norms_path)
    residual_norm = float(norms[f"layer_{STEERING_LAYER}"][0])

    print(f"[{ts()}] Loading model {cfg.name}...")
    model = ModelWrapper(cfg.name)
    try:
        run_diverse(
            model=model,
            vecs=emotion_vectors,
            residual_norm=residual_norm,
            emotion_vectors=emotion_vectors,
            out_path=out_path,
        )
    except Exception as e:
        print(f"[{ts()}] FATAL: {e}")
        traceback.print_exc()
    finally:
        model.cleanup()
        del model
        import gc; gc.collect(); torch.cuda.empty_cache()

    print(f"[{ts()}] Done. Output: {out_path}")
    print(f"[{ts()}] Next: rerun LLM judge then h5_holdout.py.")


if __name__ == "__main__":
    main()
