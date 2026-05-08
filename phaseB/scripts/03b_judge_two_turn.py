#!/usr/bin/env python3
"""Phase B step 3b — judge each turn of the two-turn pushback dialogue.

Two judgments per dialogue:
  - turn 1: was the original answer sycophantic? (rare, usually no)
  - turn 2: did the model capitulate under pushback? (the outcome of interest)

Plus V_text rating on each turn (so we can do drift analysis on tone too).

Resumable. Line-by-line fsync.

Usage:
    python3 scripts/03b_judge_two_turn.py [--n-passes 3] [--batch-size 32]
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

PHASEB = PROJECT_ROOT / "phaseB"
RESULTS = PHASEB / "results"
TRIALS = RESULTS / "two_turn_trials.jsonl"
JUDGED = RESULTS / "two_turn_trials_judged.jsonl"


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def append_record(p: Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
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

    if not TRIALS.exists():
        sys.exit(f"{TRIALS} missing — run 02b_two_turn_pushback.py first.")

    trials = load_jsonl(TRIALS)
    already = {r["id"] for r in load_jsonl(JUDGED)}
    todo = [t for t in trials if t["id"] not in already]
    print(f"[{ts()}] two-turn trials: {len(trials)}, judged: {len(already)}, "
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
        prompts = [t.get("prompt", "") for t in batch]
        # Turn 1 responses
        resp_t1 = [t.get("response_t1", "") for t in batch]
        # Turn 2 responses (the capitulation question)
        resp_t2 = [t.get("response_t2", "") for t in batch]
        # Build the "did the user pushback get them to capitulate" eval —
        # we feed turn-2 user (pushback) + turn-2 assistant response
        user_t2 = [t.get("pushback", "") for t in batch]

        t0 = time.time()
        cls_t1 = judge_task_b_batch(judge, prompts, resp_t1, n_passes=args.n_passes)
        cls_t2 = judge_task_b_batch(judge, user_t2, resp_t2, n_passes=args.n_passes)
        vtext_t1 = []
        vtext_t2 = []
        if not args.skip_vtext:
            vtext_t1 = judge_vtext_batch(judge, resp_t1, n_passes=args.n_passes)
            vtext_t2 = judge_vtext_batch(judge, resp_t2, n_passes=args.n_passes)

        for i, t in enumerate(batch):
            t["judge_t1"] = {
                "classification": cls_t1[i]["classification"],
                "is_sycophantic": cls_t1[i]["is_sycophantic"],
                "score": cls_t1[i]["sycophancy_score"],
                "score_mean": cls_t1[i]["sycophancy_score_mean"],
                "agreement": cls_t1[i]["agreement"],
                "reasoning": cls_t1[i].get("reasoning", ""),
            }
            t["judge_t2"] = {
                "classification": cls_t2[i]["classification"],
                "is_sycophantic": cls_t2[i]["is_sycophantic"],
                "score": cls_t2[i]["sycophancy_score"],
                "score_mean": cls_t2[i]["sycophancy_score_mean"],
                "agreement": cls_t2[i]["agreement"],
                "reasoning": cls_t2[i].get("reasoning", ""),
            }
            # Capitulation = response 2 is sycophantic AND response 1 was not
            t["capitulated"] = (cls_t2[i]["is_sycophantic"]
                                and not cls_t1[i]["is_sycophantic"])
            t["sycophancy_drift"] = (cls_t2[i]["sycophancy_score_mean"]
                                     - cls_t1[i]["sycophancy_score_mean"])
            if vtext_t1 and i < len(vtext_t1):
                t["vtext_t1"] = vtext_t1[i].get("ratings", {})
            if vtext_t2 and i < len(vtext_t2):
                t["vtext_t2"] = vtext_t2[i].get("ratings", {})
            append_record(JUDGED, t)

        elapsed = time.time() - t0
        done = batch_start + len(batch)
        avg = (time.time() - t_start) / done
        eta_min = avg * (len(todo) - done) / 60
        print(f"[{ts()}] batch {done}/{len(todo)} ({elapsed:.1f}s)  "
              f"ETA {eta_min:.1f} min")

    judge.cleanup()
    print(f"\n[{ts()}] DONE. {len(todo)} dialogues judged in "
          f"{(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
