#!/usr/bin/env python3
"""Layer sweep: re-extract V_internal probes at multiple layers for the 80
Phase 4 extended_unsteered trials, by forward-passing prompt+response through
Llama 3.1 70B and reading residual-stream activations at each target layer.

This is forward-only (no generation, no steering), so deterministic and fast:
~80 forward passes of ~600-token sequences. Estimated wall time on 8xA100:
~1-3 minutes after model load.

Output: results/phase4/llama70b/extended_unsteered_layer_sweep.jsonl
Each row contains:
  key, task_id, judge_classification, vtext_ratings (carried over),
  emotion_probes_per_layer: {layer_idx: {emotion_name: probe}},
  original_layer53_probes (for sanity check vs in-file values)
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODELS, EMOTION_SUBSET_50, TASK_A_VARIANTS  # noqa: E402

LAYERS = [13, 26, 39, 52, 53, 65]
TOKEN_OFFSET = 50
MODEL_KEY = "llama-70b"
INPUT_JSONL = PROJECT_ROOT / "results/phase4/llama70b/extended_unsteered_judged.jsonl"
OUTPUT_JSONL = PROJECT_ROOT / "results/phase4/llama70b/extended_unsteered_layer_sweep.jsonl"
VECTORS_DIR = PROJECT_ROOT / "data/phase1/llama70b/vectors"


def format_chat_prompt(tokenizer, system_msg: str, user_msg: str) -> str:
    """Match the prompt format used in phase2_steering.py."""
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        parts = "<|begin_of_text|>"
        if system_msg:
            parts += f"<|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>"
        parts += f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
        parts += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return parts


def compute_probes(activations: torch.Tensor, emotion_vectors: dict,
                   token_offset: int = 50) -> dict:
    """Match compute_emotion_probes() in phase2_steering.py exactly."""
    seq_len = activations.shape[1]
    start = min(token_offset, seq_len - 1)
    if start >= seq_len - 1:
        start = 0
    mean_act = activations[0, start:, :].mean(dim=0).float().cpu().numpy()
    probes = {}
    for emo in EMOTION_SUBSET_50:
        if emo in emotion_vectors:
            vec = emotion_vectors[emo].astype(np.float32)
            probes[emo] = float(np.dot(mean_act, vec) / (np.linalg.norm(vec) + 1e-10))
    return probes


def main():
    if not INPUT_JSONL.exists():
        print(f"FATAL: missing input {INPUT_JSONL}")
        sys.exit(2)
    # Load trials
    trials = [json.loads(line) for line in open(INPUT_JSONL) if line.strip()]
    print(f"Loaded {len(trials)} trials from {INPUT_JSONL.name}")

    # Resume support
    done_keys = set()
    if OUTPUT_JSONL.exists():
        with open(OUTPUT_JSONL) as f:
            for line in f:
                try:
                    done_keys.add(json.loads(line)["key"])
                except Exception:
                    pass
    todo = [t for t in trials if t["key"] not in done_keys]
    print(f"Already done: {len(done_keys)}.  To do: {len(todo)}.")

    # Load emotion vectors per layer
    emo_vecs = {}
    for L in LAYERS:
        path = VECTORS_DIR / f"emotion_vectors_layer_{L}.npz"
        emo_vecs[L] = np.load(path)
        print(f"  layer {L} vectors: {len(emo_vecs[L].files)} emotions, dim={emo_vecs[L][emo_vecs[L].files[0]].shape[0]}")

    # Load model
    from src.model import ModelWrapper
    cfg = MODELS[MODEL_KEY]
    print(f"Loading {cfg.name} ...")
    wrapper = ModelWrapper(cfg.name)
    model = wrapper.model
    tokenizer = wrapper.tokenizer
    layers_module = wrapper.layers
    model.eval()

    # Variant lookup
    variants = {v["id"]: v for v in TASK_A_VARIANTS}

    # Register forward hooks at each target layer
    captured = {}
    def make_hook(idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[idx] = h.detach()
        return hook

    handles = []
    for L in LAYERS:
        if L >= len(layers_module):
            print(f"WARN: layer {L} >= num_layers {len(layers_module)}; skipping")
            continue
        handles.append(layers_module[L].register_forward_hook(make_hook(L)))
    print(f"Hooks registered at layers: {[L for L in LAYERS if L < len(layers_module)]}")

    # Process
    t_start = time.time()
    out_f = open(OUTPUT_JSONL, "a")
    for i, t in enumerate(todo):
        variant = variants.get(t["task_id"])
        if variant is None:
            print(f"WARN: no variant config for task_id={t['task_id']}; skipping")
            continue
        prompt_text = format_chat_prompt(tokenizer, variant["system"], variant["prompt"])
        full_text = prompt_text + (t["response"] or "")
        inputs = tokenizer(full_text, return_tensors="pt").to(next(model.parameters()).device)

        captured.clear()
        with torch.no_grad():
            model(**inputs)

        probes_per_layer = {}
        for L in LAYERS:
            if L not in captured:
                continue
            probes_per_layer[L] = compute_probes(captured[L], emo_vecs[L], TOKEN_OFFSET)

        record = {
            "key": t["key"],
            "task_id": t["task_id"],
            "emotion": t.get("emotion"),
            "strength": t.get("strength"),
            "rollout": t.get("rollout"),
            "judge_classification": t.get("judge_classification"),
            "vtext_ratings": t.get("vtext_ratings"),
            "vtext_std": t.get("vtext_std"),
            "original_layer53_emotion_probes": t.get("emotion_probes"),
            "emotion_probes_per_layer": probes_per_layer,
            "seq_len": int(inputs.input_ids.shape[1]),
        }
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()

        # Memory hygiene
        del inputs
        for L in list(captured.keys()):
            del captured[L]
        captured.clear()
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(todo) - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{len(todo)} done  elapsed={elapsed:.0f}s  rate={rate:.2f}/s  ETA={eta:.0f}s")

    out_f.close()
    for h in handles:
        h.remove()

    # Sanity check: how well does our recomputed layer-53 probe match the file?
    print("\n=== Sanity check: recomputed L53 vs stored emotion_probes['desperate'] ===")
    rs = []
    with open(OUTPUT_JSONL) as f:
        for line in f:
            r = json.loads(line)
            orig = (r.get("original_layer53_emotion_probes") or {}).get("desperate")
            new = (r["emotion_probes_per_layer"].get("53") or
                   r["emotion_probes_per_layer"].get(53) or {}).get("desperate")
            if orig is not None and new is not None:
                rs.append((orig, new))
    if rs:
        orig_a = np.array([a for a, b in rs])
        new_a = np.array([b for a, b in rs])
        corr = float(np.corrcoef(orig_a, new_a)[0, 1])
        mad = float(np.mean(np.abs(orig_a - new_a)))
        print(f"  n={len(rs)} pearson r={corr:+.4f}  mean abs diff={mad:.4f}")
        print(f"  original L53 desperate range: [{orig_a.min():.3f}, {orig_a.max():.3f}]")
        print(f"  recomputed L53 desperate range: [{new_a.min():.3f}, {new_a.max():.3f}]")

    print(f"\nDONE. Wrote {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
