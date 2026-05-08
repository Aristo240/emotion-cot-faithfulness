#!/usr/bin/env python3
"""Phase B step 3 — apply the Qwen 2.5 72B judge to each Llama response in
phaseB/results/trials.jsonl, producing phaseB/results/trials_judged.jsonl.

Two judge outputs per response:
  - judge_classification: 5-point sycophancy scale (STRONG_PUSHBACK ...
    STRONG_SYCOPHANCY) — uses src.judge.judge_task_b_batch
  - vtext_ratings: 8-dim emotional tone (V_text) — uses judge_vtext_batch

Resumable: skips entries with matching `id` already in trials_judged.jsonl.
Each record is fsync'd line-by-line.

Usage:
    python3 scripts/03_judge_responses.py [--n-passes 3] [--batch-size 32]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import JudgeConfig  # noqa: E402

PHASEB_ROOT = PROJECT_ROOT / "phaseB"
RESULTS_DIR = PHASEB_ROOT / "results"
TRIALS_PATH = RESULTS_DIR / "trials.jsonl"
JUDGED_PATH = RESULTS_DIR / "trials_judged.jsonl"


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_record(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-passes", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--skip-vtext", action="store_true")
    args = p.parse_args()

    if not TRIALS_PATH.exists():
        sys.exit(f"{TRIALS_PATH} not found. Run 02_run_with_probes.py first.")

    trials = load_jsonl(TRIALS_PATH)
    already = {r["id"] for r in load_jsonl(JUDGED_PATH)}
    todo = [t for t in trials if t["id"] not in already]
    print(f"[{ts()}] trials.jsonl: {len(trials)}, already judged: {len(already)}, "
          f"remaining: {len(todo)}")
    if not todo:
        return

    from src.judge import JudgeModel, judge_task_b_batch, judge_vtext_batch  # noqa: E402

    cfg = JudgeConfig(n_passes=args.n_passes, batch_size=args.batch_size)
    judge = JudgeModel(cfg)
    print(f"[{ts()}] loading judge ({cfg.model_name})...")
    judge.load()

    t_start = time.time()
    for batch_start in range(0, len(todo), args.batch_size):
        batch = todo[batch_start: batch_start + args.batch_size]
        responses = [t.get("response", "") for t in batch]
        user_prompts = [t.get("prompt", "") for t in batch]
        t0 = time.time()
        cls = judge_task_b_batch(judge, user_prompts, responses, n_passes=args.n_passes)
        vtext = []
        if not args.skip_vtext:
            vtext = judge_vtext_batch(judge, responses, n_passes=args.n_passes)

        for i, t in enumerate(batch):
            t["judge_classification"] = cls[i]["classification"]
            t["judge_classifications_per_pass"] = cls[i]["classifications_per_pass"]
            t["judge_agreement"] = cls[i]["agreement"]
            t["judge_is_sycophantic"] = cls[i]["is_sycophantic"]
            t["judge_sycophancy_score"] = cls[i]["sycophancy_score"]
            t["judge_sycophancy_score_mean"] = cls[i]["sycophancy_score_mean"]
            t["judge_reasoning"] = cls[i].get("reasoning", "")
            if vtext and i < len(vtext):
                t["vtext_ratings"] = vtext[i].get("ratings", {})
                t["vtext_std"] = vtext[i].get("std_per_dimension", {})
                t["vtext_ratings_per_pass"] = vtext[i].get("ratings_per_pass", [])
            append_record(JUDGED_PATH, t)

        elapsed = time.time() - t0
        done = batch_start + len(batch)
        avg = (time.time() - t_start) / done
        eta_min = avg * (len(todo) - done) / 60
        print(f"[{ts()}] batch {done}/{len(todo)} ({elapsed:.1f}s)   "
              f"ETA {eta_min:.1f} min")

    judge.cleanup()
    print(f"\n[{ts()}] DONE. {len(todo)} judged in "
          f"{(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
