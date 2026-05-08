#!/usr/bin/env python3
"""Phase B step 2 — for each prompt in the sycophancy benchmark:
  (1) tokenise the prompt;
  (2) forward pass with a hook on layer L → record V_internal at the LAST USER
      TOKEN as the cosine of the residual against each of the 50 emotion vectors;
  (3) generate a response (greedy or sampled) and record the text.

The output is a flat JSONL at phaseB/results/trials.jsonl, one record per prompt:

    {"id": "...", "category": "...", "prompt": "...",
     "response": "...", "emotion_probes": {"happy": 0.42, "sad": -0.31, ...},
     "elapsed_s": 9.7, "timestamp": "2026-05-09 03:14:15"}

The script is resumable: re-running picks up at the first `id` not already in
the output jsonl. Each record is fsync'd line-by-line.

Usage:
    python3 scripts/02_run_with_probes.py                   # defaults
    python3 scripts/02_run_with_probes.py --layer 53 --max-new-tokens 384

Notes for Sunday morning:
  - Llama 3.1 70B fp16 needs all 8 V100s. Reuses the existing project's
    ModelWrapper (../src/model.py) which already does pipeline parallel loading.
  - HF_TOKEN must be exported (the model is gated). The .env in the project
    root has it; `set -a; source ../.env; set +a` before invoking.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Project layout: phaseB/scripts/ -> project root is two levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODELS  # noqa: E402
from src.model import ModelWrapper  # noqa: E402

LLAMA_70B_MODEL_NAME = MODELS["llama-70b"].name

PHASEB_ROOT = PROJECT_ROOT / "phaseB"
DATA_DIR = PHASEB_ROOT / "data"
RESULTS_DIR = PHASEB_ROOT / "results"
EMOTION_VECTORS_PATH = PROJECT_ROOT / "data/phase1/llama70b/vectors/emotion_vectors_layer_{layer}.npz"


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None,
                   help="Input JSONL (default: first sycophancy_*.jsonl in data/)")
    p.add_argument("--output", default=None,
                   help="Output JSONL (default: phaseB/results/trials.jsonl)")
    p.add_argument("--layer", type=int, default=53,
                   help="Layer to read residual stream from (default: 53, "
                        "matches Phase 1 vectors)")
    p.add_argument("--max-new-tokens", type=int, default=384,
                   help="Max tokens to generate per response (default: 384).")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature; 0.0 = greedy (default: greedy "
                        "for reproducibility).")
    p.add_argument("--probe-position", default="last_user",
                   choices=["last_user", "mean_response", "mean_prompt"],
                   help="Where to read V_internal from (default: last_user "
                        "= residual at the last user token before generation).")
    p.add_argument("--limit", type=int, default=0,
                   help="Cap number of prompts (0 = all).")
    return p.parse_args()


def find_default_input() -> Path:
    candidates = sorted(DATA_DIR.glob("sycophancy_*.jsonl"))
    if not candidates:
        sys.exit(f"No sycophancy_*.jsonl in {DATA_DIR}. Run 01_download_data.py first.")
    return candidates[0]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_emotion_vectors(layer: int) -> dict[str, np.ndarray]:
    p = Path(str(EMOTION_VECTORS_PATH).format(layer=layer))
    if not p.exists():
        sys.exit(f"Emotion vectors not found at {p}. "
                 f"Run scripts/01_run_phase1.py first or pass --layer with vectors available.")
    npz = np.load(p, allow_pickle=True)
    vecs = {k: np.asarray(npz[k], dtype=np.float32) for k in npz.files}
    print(f"[{ts()}] loaded {len(vecs)} emotion vectors at layer {layer} "
          f"(d={next(iter(vecs.values())).shape[0]})")
    return vecs


def cosine_against_vectors(residual: np.ndarray,
                           vectors: dict[str, np.ndarray]) -> dict[str, float]:
    """residual: (hidden_dim,) → returns {emotion: cosine}."""
    r_norm = np.linalg.norm(residual) + 1e-9
    out = {}
    for em, v in vectors.items():
        out[em] = float(np.dot(residual, v) / (r_norm * (np.linalg.norm(v) + 1e-9)))
    return out


def append_record(out_path: Path, rec: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def build_prompt(model: ModelWrapper, user_text: str) -> str:
    """Apply the chat template so the prompt ends at the assistant header.

    The "last user token" probe position is then unambiguous: it's the last
    token of the user turn, just before the model is about to speak. This
    matches the Sofroniew assistant-header position.
    """
    messages = [{"role": "user", "content": user_text}]
    try:
        return model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        # Fallback if chat template missing
        return f"User: {user_text}\nAssistant:"


def main():
    args = parse_args()
    in_path = Path(args.input) if args.input else find_default_input()
    out_path = Path(args.output) if args.output else RESULTS_DIR / "trials.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = load_jsonl(in_path)
    if not items:
        sys.exit(f"No items in {in_path}")
    if args.limit > 0:
        items = items[: args.limit]

    already = {r["id"] for r in load_jsonl(out_path)}
    todo = [r for r in items if r["id"] not in already]
    print(f"[{ts()}] input: {in_path.name} ({len(items)} items)")
    print(f"[{ts()}] output: {out_path.name} ({len(already)} already done, "
          f"{len(todo)} remaining)")
    if not todo:
        print(f"[{ts()}] All done.")
        return

    vectors = load_emotion_vectors(args.layer)
    print(f"[{ts()}] loading {LLAMA_70B_MODEL_NAME} (this takes ~5-10 min on 8x V100)...")
    model = ModelWrapper(LLAMA_70B_MODEL_NAME)
    print(f"[{ts()}] model loaded: {model.num_layers} layers, hidden={model.hidden_dim}")

    if args.layer >= model.num_layers:
        sys.exit(f"--layer {args.layer} but model has only {model.num_layers} layers")

    t_start = time.time()
    for i, rec in enumerate(todo):
        t0 = time.time()
        prompt_text = build_prompt(model, rec["prompt"])

        # Step (a) — forward pass to extract residual at the last token
        # We do this BEFORE generation so the probe is causally
        # upstream of the response. ModelWrapper.extract_activations
        # already runs a no-grad forward and returns CPU tensors.
        acts = model.extract_activations(prompt_text, layer_indices=[args.layer])
        # acts[layer] shape: (1, seq_len, hidden_dim)
        residual = acts[args.layer]
        if args.probe_position == "last_user":
            vec = residual[0, -1, :].detach().cpu().float().numpy()
        elif args.probe_position == "mean_prompt":
            vec = residual[0].mean(dim=0).detach().cpu().float().numpy()
        else:
            # mean_response only makes sense AFTER generation; fall back to last_user here
            vec = residual[0, -1, :].detach().cpu().float().numpy()
        probes = cosine_against_vectors(vec, vectors)

        # Step (b) — generate the response (no steering)
        gen_out = model.generate_steered(
            prompt=prompt_text,
            steering_configs=None,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=0.95,
            do_sample=args.temperature > 0,
        )
        response = gen_out["text"]

        out_rec = {
            "id": rec["id"],
            "category": rec.get("category"),
            "split": rec.get("split"),
            "prompt": rec["prompt"],
            "response": response,
            "emotion_probes": probes,
            "probe_layer": args.layer,
            "probe_position": args.probe_position,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "elapsed_s": round(time.time() - t0, 2),
            "timestamp": ts(),
        }
        append_record(out_path, out_rec)

        if (i + 1) % 10 == 0 or i < 3:
            elapsed = time.time() - t_start
            avg = elapsed / (i + 1)
            eta_min = avg * (len(todo) - i - 1) / 60
            print(f"[{ts()}] [{i+1}/{len(todo)}] {rec['id']}  "
                  f"({out_rec['elapsed_s']}s)  "
                  f"V_int[desperate]={probes.get('desperate', 0):+.3f} "
                  f"V_int[calm]={probes.get('calm', 0):+.3f}  "
                  f"ETA {eta_min:.0f} min")

    print(f"\n[{ts()}] DONE. Wrote {len(todo)} new trials in "
          f"{(time.time() - t_start)/60:.1f} min to {out_path}")


if __name__ == "__main__":
    main()
