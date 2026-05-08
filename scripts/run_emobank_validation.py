#!/usr/bin/env python3
"""EmoBank PAD external validation — chain-after-judge job for the Lambda run.

For each EmoBank sentence:
  - Forward pass through Llama 3.1 70B with a hook on layer 53.
  - Read the residual at the last token (assistant-header position).
  - Cosine against each of our 50 emotion vectors → V_internal feature.
  - Record the EmoBank Valence/Arousal/Dominance ratings as ground truth.

Then the analysis (script-end):
  - Linear regression: V_internal (50d) → V/A/D ratings, 5-fold CV R^2.
  - Per-vector cosine vs each rating → which V_internal axes line up with PAD?

Output:
  results/emobank/llama70b/trials.jsonl       (one record per sentence)
  results/emobank/llama70b/report.json        (R^2, top-loading vectors)

This is the cheapest external-validity check we can run: same model, same
hooks, same 50 vectors as Phase 1 — no behavioural elicitation, no judge.

Resumable: re-running picks up at the first id not in trials.jsonl.

Usage:
    python3 scripts/run_emobank_validation.py [--limit 0] [--layer 53]
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODELS  # noqa: E402
from src.model import ModelWrapper  # noqa: E402

LLAMA = MODELS["llama-70b"].name
EMOTION_VECTORS_PATH = PROJECT_ROOT / "data/phase1/llama70b/vectors/emotion_vectors_layer_{layer}.npz"
OUT_DIR = PROJECT_ROOT / "results/emobank/llama70b"
TRIALS_PATH = OUT_DIR / "trials.jsonl"
REPORT_PATH = OUT_DIR / "report.json"


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_record(p: Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=53)
    p.add_argument("--limit", type=int, default=0,
                   help="Cap number of sentences (0 = all ~10k).")
    return p.parse_args()


def load_emobank() -> list[dict]:
    """Try several public sources of EmoBank, in order of reliability."""
    # Source 1: HuggingFace mirror SocialEmoCorpus/emobank
    try:
        from datasets import load_dataset
        ds = load_dataset("SocialEmoCorpus/emobank", split="train")
        print(f"[{ts()}] loaded EmoBank from SocialEmoCorpus/emobank: {len(ds)} rows")
        rows = []
        for i, r in enumerate(ds):
            text = r.get("text") or r.get("sentence") or r.get("Text", "")
            v = r.get("V") or r.get("valence")
            a = r.get("A") or r.get("arousal")
            d = r.get("D") or r.get("dominance")
            if text and v is not None and a is not None and d is not None:
                rows.append({"id": f"eb_{i:06d}", "text": text,
                             "V": float(v), "A": float(a), "D": float(d)})
        if rows:
            return rows
    except Exception as e:
        print(f"[{ts()}] SocialEmoCorpus/emobank failed: {e}")

    # Source 2: raw CSV from JULIELab/EmoBank GitHub
    import urllib.request
    import csv
    import io
    url = "https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/emobank.csv"
    try:
        print(f"[{ts()}] trying JULIELab/EmoBank CSV: {url}")
        with urllib.request.urlopen(url, timeout=30) as resp:
            text = resp.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        rows = []
        for i, r in enumerate(reader):
            try:
                rows.append({
                    "id": r.get("id") or f"eb_{i:06d}",
                    "text": r["text"],
                    "V": float(r["V"]),
                    "A": float(r["A"]),
                    "D": float(r["D"]),
                    "split": r.get("split", "")
                })
            except (KeyError, ValueError):
                continue
        print(f"[{ts()}] loaded {len(rows)} rows from JULIELab/EmoBank")
        return rows
    except Exception as e:
        sys.exit(f"Cannot load EmoBank from any source. Last error: {e}")


def cosine_against_vectors(residual: np.ndarray,
                           vectors: dict[str, np.ndarray]) -> dict[str, float]:
    r_norm = float(np.linalg.norm(residual)) + 1e-9
    out = {}
    for em, v in vectors.items():
        out[em] = float(np.dot(residual, v) / (r_norm * (float(np.linalg.norm(v)) + 1e-9)))
    return out


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_emobank()
    if args.limit > 0:
        rows = rows[: args.limit]
    already = {r["id"] for r in load_jsonl(TRIALS_PATH)}
    todo = [r for r in rows if r["id"] not in already]
    print(f"[{ts()}] EmoBank rows: {len(rows)}, already done: {len(already)}, "
          f"remaining: {len(todo)}")

    if not todo:
        print(f"[{ts()}] All sentences already processed. Running analysis only.")
    else:
        vec_path = Path(str(EMOTION_VECTORS_PATH).format(layer=args.layer))
        npz = np.load(vec_path, allow_pickle=True)
        vectors = {k: np.asarray(npz[k], dtype=np.float32) for k in npz.files}
        print(f"[{ts()}] loaded {len(vectors)} emotion vectors from {vec_path.name}")

        print(f"[{ts()}] loading {LLAMA} (5-10 min)...")
        model = ModelWrapper(LLAMA)
        print(f"[{ts()}] model: {model.num_layers} layers, hidden={model.hidden_dim}")
        if args.layer >= model.num_layers:
            sys.exit(f"--layer {args.layer} >= num_layers {model.num_layers}")

        t_start = time.time()
        for i, row in enumerate(todo):
            t0 = time.time()
            text = row["text"]
            # No chat template — these are stand-alone sentences. We just
            # tokenise the raw text and read the last-token residual.
            acts = model.extract_activations(text, layer_indices=[args.layer])
            residual = acts[args.layer][0, -1, :].detach().cpu().float().numpy()
            probes = cosine_against_vectors(residual, vectors)
            rec = {
                "id": row["id"],
                "text": text,
                "V": row["V"], "A": row["A"], "D": row["D"],
                "split": row.get("split", ""),
                "emotion_probes": probes,
                "probe_layer": args.layer,
                "elapsed_s": round(time.time() - t0, 3),
                "timestamp": ts(),
            }
            append_record(TRIALS_PATH, rec)
            if (i + 1) % 50 == 0 or i < 3:
                avg = (time.time() - t_start) / (i + 1)
                eta_min = avg * (len(todo) - i - 1) / 60
                print(f"[{ts()}] [{i+1}/{len(todo)}] {row['id']}  "
                      f"({rec['elapsed_s']}s)  "
                      f"V_int[desperate]={probes.get('desperate',0):+.3f}  "
                      f"V_int[calm]={probes.get('calm',0):+.3f}  "
                      f"ETA {eta_min:.1f} min")

        print(f"\n[{ts()}] forward pass DONE in {(time.time()-t_start)/60:.1f} min")
        model.cleanup()

    # ---------- analysis ----------
    print(f"\n[{ts()}] running analysis ...")
    data = load_jsonl(TRIALS_PATH)
    if not data:
        sys.exit("No trials yet — nothing to analyse.")

    emo_names = list(data[0]["emotion_probes"].keys())
    X = np.array([[r["emotion_probes"][e] for e in emo_names] for r in data])
    Y = np.array([[r["V"], r["A"], r["D"]] for r in data])
    print(f"[{ts()}] design matrix: {X.shape}  targets: {Y.shape}")

    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr

    # 5-fold CV R^2 per dimension
    kf = KFold(n_splits=5, shuffle=True, random_state=20260510)
    cv_r2 = {dim: [] for dim in ["V", "A", "D"]}
    for tr, te in kf.split(X):
        for j, dim in enumerate(["V", "A", "D"]):
            m = RidgeCV(alphas=[0.1, 1, 10, 100]).fit(X[tr], Y[tr, j])
            cv_r2[dim].append(r2_score(Y[te, j], m.predict(X[te])))
    cv_r2_mean = {dim: float(np.mean(scores)) for dim, scores in cv_r2.items()}
    cv_r2_std = {dim: float(np.std(scores)) for dim, scores in cv_r2.items()}
    print(f"[{ts()}] 5-fold CV R^2 — V: {cv_r2_mean['V']:.3f} ± {cv_r2_std['V']:.3f}, "
          f"A: {cv_r2_mean['A']:.3f} ± {cv_r2_std['A']:.3f}, "
          f"D: {cv_r2_mean['D']:.3f} ± {cv_r2_std['D']:.3f}")

    # Per-vector Pearson r against V/A/D
    per_vec = {}
    for j, dim in enumerate(["V", "A", "D"]):
        rs = []
        for k, em in enumerate(emo_names):
            r, p = pearsonr(X[:, k], Y[:, j])
            rs.append((em, float(r), float(p)))
        rs.sort(key=lambda t: abs(t[1]), reverse=True)
        per_vec[dim] = rs
        print(f"\n[{ts()}] top |r| with {dim}:")
        for em, r, p in rs[:8]:
            print(f"    {em:>14s}  r={r:+.3f}  p={p:.2g}")

    report = {
        "n_sentences": int(X.shape[0]),
        "n_vectors": int(X.shape[1]),
        "probe_layer": int(data[0]["probe_layer"]),
        "cv_r2_mean": cv_r2_mean,
        "cv_r2_std": cv_r2_std,
        "per_dim_top_correlations": {
            dim: [{"emotion": e, "r": r, "p": p} for e, r, p in rs[:15]]
            for dim, rs in per_vec.items()
        },
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[{ts()}] wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
