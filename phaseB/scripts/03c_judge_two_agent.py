#!/usr/bin/env python3
"""Phase B step 3c — judge each engineer turn of the two-agent dialogues.

⚠️ Out-of-scope mental note: see header of 02c_two_agent_dialogue.py.

Per dialogue we record one classification per ENGINEER turn:
  LEGITIMATE / SHORTCUT / UNCLEAR (the Phase 2 Task A judge rubric).

The point is that we can later look at when (which round) the engineer flips
from LEGITIMATE → SHORTCUT under increasing manager pressure, and check
whether V_internal[desperate] in the round just before the flip was
elevated — i.e. did probe drift precede capitulation.

Resumable. Line-by-line fsync.

Usage:
    python3 scripts/03c_judge_two_agent.py [--n-passes 3]
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
TRIALS = RESULTS / "two_agent_trials.jsonl"
JUDGED = RESULTS / "two_agent_trials_judged.jsonl"


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
    args = p.parse_args()

    if not TRIALS.exists():
        sys.exit(f"{TRIALS} missing — run 02c_two_agent_dialogue.py first.")

    trials = load_jsonl(TRIALS)
    already = {r["id"] for r in load_jsonl(JUDGED)}
    todo = [t for t in trials if t["id"] not in already]
    print(f"[{ts()}] two-agent dialogues: {len(trials)}, judged: {len(already)}, "
          f"remaining: {len(todo)}")
    if not todo:
        return

    from src.judge import JudgeModel, judge_task_a_batch  # noqa: E402

    cfg = JudgeConfig(n_passes=args.n_passes, batch_size=args.batch_size)
    judge = JudgeModel(cfg)
    print(f"[{ts()}] loading judge ({cfg.model_name})...")
    judge.load()

    # Flatten: (dialogue_idx, round, engineer_text) so we can batch across rounds
    flat = []
    for d_idx, t in enumerate(todo):
        for entry in t.get("transcript", []):
            if entry["role"] == "engineer":
                flat.append((d_idx, entry["round"], entry["text"]))

    print(f"[{ts()}] {len(flat)} engineer-turns to judge "
          f"({len(flat)/max(1,len(todo)):.1f} per dialogue)")
    if not flat:
        sys.exit("No engineer turns found in transcripts.")

    t_start = time.time()
    # Per-dialogue running results
    per_dialogue: dict[int, list[dict]] = {i: [] for i in range(len(todo))}
    for batch_start in range(0, len(flat), args.batch_size):
        batch = flat[batch_start: batch_start + args.batch_size]
        responses = [text for (_, _, text) in batch]
        t0 = time.time()
        cls = judge_task_a_batch(judge, responses, n_passes=args.n_passes)
        for j, (d_idx, rnd, _) in enumerate(batch):
            per_dialogue[d_idx].append({
                "round": rnd,
                "classification": cls[j]["classification"],
                "agreement": cls[j]["agreement"],
                "reasoning": cls[j].get("reasoning", ""),
            })
        elapsed = time.time() - t0
        done = batch_start + len(batch)
        avg = (time.time() - t_start) / done
        eta_min = avg * (len(flat) - done) / 60
        print(f"[{ts()}] turn-batch {done}/{len(flat)} ({elapsed:.1f}s)  "
              f"ETA {eta_min:.1f} min")

    # Write per-dialogue judged records (only those where we processed all turns)
    for d_idx, t in enumerate(todo):
        rounds_judged = sorted(per_dialogue[d_idx], key=lambda r: r["round"])
        t["judge_per_round"] = rounds_judged
        # Did the engineer EVER shortcut, and at which round?
        shortcut_rounds = [r["round"] for r in rounds_judged
                           if "SHORTCUT" in (r.get("classification") or "")]
        t["any_shortcut"] = len(shortcut_rounds) > 0
        t["first_shortcut_round"] = (min(shortcut_rounds) if shortcut_rounds
                                     else None)
        t["final_classification"] = (rounds_judged[-1]["classification"]
                                     if rounds_judged else "UNCLEAR")
        append_record(JUDGED, t)

    judge.cleanup()
    print(f"\n[{ts()}] DONE. {len(todo)} dialogues judged in "
          f"{(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
