#!/usr/bin/env python3
"""Re-classify Phase 4 trials with the Qwen 2.5 72B judge.

Phase 4 jsonls are all Task A (reward-hacking) format and only contain regex
classifications. This script applies the same judge pipeline that was used on
Phase 2 to all 4 Phase 4 blocker outputs:

    finegrained.jsonl
    extended_unsteered.jsonl
    random_directions.jsonl
    text_injection.jsonl

For each input file `X.jsonl` it produces `X_judged.jsonl` alongside, with
appended fields: `judge_classification`, `judge_classifications_per_pass`,
`judge_agreement`, `judge_reasoning`, `vtext_ratings`, `vtext_std`,
`vtext_ratings_per_pass`.

Resumable: if `X_judged.jsonl` already contains entries with matching `key`
they are skipped. Each judged record is fsync'd line-by-line so an interrupted
run can be re-launched without losing progress.

Usage:
    python scripts/run_judge_phase4.py [--batch-size 32] [--n-passes 3]
                                       [--skip-vtext]
                                       [--only finegrained,extended_unsteered]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import JudgeConfig  # noqa: E402

PHASE4_REL_PATHS = [
    "results/phase4/llama70b/finegrained.jsonl",
    "results/phase4/llama70b/extended_unsteered.jsonl",
    "results/phase4/llama70b/random_directions.jsonl",
    "results/phase4/llama70b/text_injection.jsonl",
]


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def append_jsonl(path: Path, record: dict) -> None:
    """Append one record and fsync so partial runs are recoverable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        try:
            import os
            os.fsync(f.fileno())
        except (OSError, AttributeError):
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-passes", type=int, default=3)
    p.add_argument("--skip-vtext", action="store_true",
                   help="Skip V_text 8-dim rating (only run classification).")
    p.add_argument("--only", type=str, default="",
                   help="Comma-separated subset of file stems to process "
                        "(e.g. 'finegrained,extended_unsteered'). Empty = all.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    files: list[tuple[Path, Path]] = []
    for rel in PHASE4_REL_PATHS:
        in_path = PROJECT_ROOT / rel
        if only and in_path.stem not in only:
            continue
        out_path = in_path.with_name(f"{in_path.stem}_judged{in_path.suffix}")
        files.append((in_path, out_path))

    # First pass: count work
    plan: list[tuple[Path, Path, list[dict]]] = []
    total_todo = 0
    for in_path, out_path in files:
        entries = load_jsonl(in_path)
        already = {e.get("key") for e in load_jsonl(out_path)}
        todo = [e for e in entries if e.get("key") not in already]
        plan.append((in_path, out_path, todo))
        total_todo += len(todo)
        print(f"[{ts()}] {in_path.name}: {len(entries)} total, "
              f"{len(already)} already judged, {len(todo)} remaining")

    if total_todo == 0:
        print(f"[{ts()}] Nothing to do. All entries already judged.")
        return

    print(f"[{ts()}] TOTAL TO JUDGE: {total_todo} trials "
          f"× {args.n_passes} passes × {2 if not args.skip_vtext else 1} prompts "
          f"= {total_todo * args.n_passes * (2 if not args.skip_vtext else 1)} generations")

    # Lazy-import judge so we can dry-run argument parsing without GPU deps.
    from src.judge import JudgeModel, judge_task_a_batch, judge_vtext_batch  # noqa: E402

    cfg = JudgeConfig(n_passes=args.n_passes, batch_size=args.batch_size)
    judge = JudgeModel(cfg)
    print(f"[{ts()}] Loading judge model ({cfg.model_name})...")
    judge.load()
    print(f"[{ts()}] Judge ready.")

    processed = 0
    t_start = time.time()
    for in_path, out_path, todo in plan:
        if not todo:
            continue
        print(f"\n[{ts()}] === {in_path.name}: judging {len(todo)} entries ===")
        for batch_start in range(0, len(todo), args.batch_size):
            batch = todo[batch_start:batch_start + args.batch_size]
            responses = [e.get("response", "") for e in batch]

            t0 = time.time()
            cls_results = judge_task_a_batch(judge, responses, n_passes=args.n_passes)
            vtext_results: list[dict] = []
            if not args.skip_vtext:
                vtext_results = judge_vtext_batch(judge, responses, n_passes=args.n_passes)

            for i, entry in enumerate(batch):
                entry["judge_classification"] = cls_results[i]["classification"]
                entry["judge_classifications_per_pass"] = cls_results[i]["classifications_per_pass"]
                entry["judge_agreement"] = cls_results[i]["agreement"]
                entry["judge_reasoning"] = cls_results[i].get("reasoning", "")
                if vtext_results and i < len(vtext_results):
                    entry["vtext_ratings"] = vtext_results[i].get("ratings", {})
                    entry["vtext_std"] = vtext_results[i].get("std_per_dimension", {})
                    entry["vtext_ratings_per_pass"] = vtext_results[i].get("ratings_per_pass", [])
                append_jsonl(out_path, entry)

            processed += len(batch)
            elapsed = time.time() - t0
            avg = (time.time() - t_start) / processed
            eta_min = avg * (total_todo - processed) / 60
            print(f"[{ts()}]   batch {batch_start + len(batch):>4d}/{len(todo)} "
                  f"({elapsed:5.1f}s)   total {processed}/{total_todo}   "
                  f"ETA {eta_min:5.1f} min")

    judge.cleanup()
    elapsed_total = (time.time() - t_start) / 60
    print(f"\n[{ts()}] DONE. Judged {processed} trials in {elapsed_total:.1f} min.")
    for _, out_path, _ in plan:
        if out_path.exists():
            n = sum(1 for _ in open(out_path))
            print(f"[{ts()}]   {out_path.name}: {n} judged records")


if __name__ == "__main__":
    main()
