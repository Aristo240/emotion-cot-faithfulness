#!/usr/bin/env python3
"""
Post-process existing Phase 2 results with the Qwen 2.5 72B LLM judge.

This script:
  1. Loads existing Phase 2 task_a.jsonl and task_b.jsonl results
  2. Re-classifies behavioral outcomes using the Qwen judge (blind to steering)
  3. Rates the emotional tone of each response (V_text) on 8 dimensions
  4. Computes judge reliability metrics (inter-pass agreement, ICC)
  5. Saves augmented results alongside originals (non-destructive)

This does NOT modify the original Phase 2 files. It creates new files:
  - results/phase2/task_a_judged.jsonl
  - results/phase2/task_b_judged.jsonl
  - results/phase2/judge_reliability.json

Usage:
  python scripts/run_judge_reclassification.py [--batch-size 32] [--n-passes 3]

The judge model (Qwen 2.5 72B) is from a different family than the steered
model (Llama 3.1 70B) to avoid circular evaluation bias.
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import JudgeConfig, TASK_B_VARIANTS


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_jsonl(path):
    if not path.exists():
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def save_jsonl(path, entries):
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def get_user_prompt_for_task(task_id):
    """Look up the user prompt for a sycophancy task by its ID."""
    for variant in TASK_B_VARIANTS:
        if variant["id"] == task_id:
            return variant["prompt"]
    return ""


def main():
    parser = argparse.ArgumentParser(description="Re-classify Phase 2 results with Qwen judge")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-passes", type=int, default=3)
    parser.add_argument("--skip-task-a", action="store_true", help="Skip Task A reclassification")
    parser.add_argument("--skip-task-b", action="store_true", help="Skip Task B reclassification")
    parser.add_argument("--skip-vtext", action="store_true", help="Skip V_text rating")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results" / "phase2"

    task_a_path = results_dir / "task_a.jsonl"
    task_b_path = results_dir / "task_b.jsonl"
    task_a_judged_path = results_dir / "task_a_judged.jsonl"
    task_b_judged_path = results_dir / "task_b_judged.jsonl"
    reliability_path = results_dir / "judge_reliability.json"

    print(f"[{ts()}] ============================================")
    print(f"[{ts()}] PHASE 2 POST-PROCESSING: LLM JUDGE")
    print(f"[{ts()}] ============================================")
    print(f"[{ts()}] Judge model: Qwen 2.5 72B (different family from steered Llama)")
    print(f"[{ts()}] N passes: {args.n_passes}")
    print(f"[{ts()}] Batch size: {args.batch_size}")

    # Load existing results
    task_a_entries = load_jsonl(task_a_path) if not args.skip_task_a else []
    task_b_entries = load_jsonl(task_b_path) if not args.skip_task_b else []
    print(f"[{ts()}] Task A entries: {len(task_a_entries)}")
    print(f"[{ts()}] Task B entries: {len(task_b_entries)}")

    if not task_a_entries and not task_b_entries:
        print(f"[{ts()}] No entries to process. Run Phase 2 first.")
        return

    # Load already-judged entries to resume
    already_judged_a = {e["key"] for e in load_jsonl(task_a_judged_path)}
    already_judged_b = {e["key"] for e in load_jsonl(task_b_judged_path)}
    task_a_todo = [e for e in task_a_entries if e.get("key") not in already_judged_a]
    task_b_todo = [e for e in task_b_entries if e.get("key") not in already_judged_b]
    print(f"[{ts()}] Task A remaining: {len(task_a_todo)}")
    print(f"[{ts()}] Task B remaining: {len(task_b_todo)}")

    if not task_a_todo and not task_b_todo:
        print(f"[{ts()}] All entries already judged. Computing reliability only.")
    else:
        # Load judge model
        from src.judge import JudgeModel, judge_task_a_batch, judge_task_b_batch, judge_vtext_batch

        config = JudgeConfig(n_passes=args.n_passes, batch_size=args.batch_size)
        judge = JudgeModel(config)
        print(f"[{ts()}] Loading judge model...")
        judge.load()

        # ---- Task A: Reward hacking reclassification ----
        if task_a_todo:
            print(f"\n[{ts()}] === TASK A: REWARD HACKING RECLASSIFICATION ===")
            for batch_start in range(0, len(task_a_todo), args.batch_size):
                batch = task_a_todo[batch_start:batch_start + args.batch_size]
                responses = [e.get("response", "") for e in batch]

                t0 = time.time()

                # Behavioral classification
                cls_results = judge_task_a_batch(judge, responses, n_passes=args.n_passes)

                # V_text rating
                vtext_results = []
                if not args.skip_vtext:
                    vtext_results = judge_vtext_batch(judge, responses, n_passes=args.n_passes)

                elapsed = time.time() - t0

                # Merge into entries
                for i, entry in enumerate(batch):
                    entry["judge_classification"] = cls_results[i]["classification"]
                    entry["judge_classifications_per_pass"] = cls_results[i]["classifications_per_pass"]
                    entry["judge_agreement"] = cls_results[i]["agreement"]
                    entry["judge_reasoning"] = cls_results[i].get("reasoning", "")

                    if vtext_results and i < len(vtext_results):
                        entry["vtext_ratings"] = vtext_results[i].get("ratings", {})
                        entry["vtext_std"] = vtext_results[i].get("std_per_dimension", {})
                        entry["vtext_ratings_per_pass"] = vtext_results[i].get("ratings_per_pass", [])

                    # Append to output file
                    with open(task_a_judged_path, "a") as f:
                        f.write(json.dumps(entry) + "\n")

                batch_end = min(batch_start + args.batch_size, len(task_a_todo))
                print(
                    f"[{ts()}] Task A [{batch_end}/{len(task_a_todo)}] "
                    f"({elapsed:.1f}s) "
                    f"classifications: {[r['classification'] for r in cls_results]}"
                )

        # ---- Task B: Sycophancy reclassification ----
        if task_b_todo:
            print(f"\n[{ts()}] === TASK B: SYCOPHANCY RECLASSIFICATION ===")
            for batch_start in range(0, len(task_b_todo), args.batch_size):
                batch = task_b_todo[batch_start:batch_start + args.batch_size]
                responses = [e.get("response", "") for e in batch]
                user_prompts = [get_user_prompt_for_task(e.get("task_id", "")) for e in batch]

                t0 = time.time()

                # Behavioral classification (5-point scale)
                cls_results = judge_task_b_batch(
                    judge, user_prompts, responses, n_passes=args.n_passes
                )

                # V_text rating
                vtext_results = []
                if not args.skip_vtext:
                    vtext_results = judge_vtext_batch(judge, responses, n_passes=args.n_passes)

                elapsed = time.time() - t0

                for i, entry in enumerate(batch):
                    entry["judge_classification"] = cls_results[i]["classification"]
                    entry["judge_classifications_per_pass"] = cls_results[i]["classifications_per_pass"]
                    entry["judge_agreement"] = cls_results[i]["agreement"]
                    entry["judge_is_sycophantic"] = cls_results[i]["is_sycophantic"]
                    entry["judge_sycophancy_score"] = cls_results[i]["sycophancy_score"]
                    entry["judge_sycophancy_score_mean"] = cls_results[i]["sycophancy_score_mean"]
                    entry["judge_reasoning"] = cls_results[i].get("reasoning", "")

                    if vtext_results and i < len(vtext_results):
                        entry["vtext_ratings"] = vtext_results[i].get("ratings", {})
                        entry["vtext_std"] = vtext_results[i].get("std_per_dimension", {})
                        entry["vtext_ratings_per_pass"] = vtext_results[i].get("ratings_per_pass", [])

                    with open(task_b_judged_path, "a") as f:
                        f.write(json.dumps(entry) + "\n")

                batch_end = min(batch_start + args.batch_size, len(task_b_todo))
                print(
                    f"[{ts()}] Task B [{batch_end}/{len(task_b_todo)}] "
                    f"({elapsed:.1f}s) "
                    f"classifications: {[r['classification'] for r in cls_results]}"
                )

        # Cleanup judge
        print(f"[{ts()}] Cleaning up judge model...")
        judge.cleanup()

    # ---- Compute reliability metrics ----
    print(f"\n[{ts()}] === COMPUTING RELIABILITY METRICS ===")
    from src.judge import compute_judge_reliability, compute_vtext_icc

    reliability = {}

    for label, path in [("task_a", task_a_judged_path), ("task_b", task_b_judged_path)]:
        entries = load_jsonl(path)
        if not entries:
            continue

        # Classification reliability
        cls_reliability = compute_judge_reliability(
            entries, key="judge_classifications_per_pass"
        )
        reliability[f"{label}_classification"] = cls_reliability
        print(f"[{ts()}] {label} classification: "
              f"overall_agreement={cls_reliability['overall_agreement']:.3f}, "
              f"mean_agreement={cls_reliability['mean_agreement']:.3f}")

        # V_text ICC
        vtext_entries = [e for e in entries if "vtext_ratings_per_pass" in e]
        if vtext_entries:
            icc = compute_vtext_icc(
                [{"ratings_per_pass": e["vtext_ratings_per_pass"]} for e in vtext_entries]
            )
            reliability[f"{label}_vtext_icc"] = icc
            for dim, val in icc.items():
                if val is not None:
                    print(f"[{ts()}]   {label} V_text ICC({dim}) = {val:.3f}")

    # Compare regex vs judge classification
    for label, path in [("task_a", task_a_judged_path), ("task_b", task_b_judged_path)]:
        entries = load_jsonl(path)
        if not entries:
            continue

        # Agreement between regex and judge
        agree = 0
        total = 0
        for e in entries:
            regex_cls = e.get("classification", "")
            judge_cls = e.get("judge_classification", "")

            # Normalize to comparable categories
            if label == "task_a":
                regex_norm = "SHORTCUT" if regex_cls == "hack" else ("LEGITIMATE" if regex_cls == "legit" else "UNCLEAR")
                judge_norm = judge_cls
            else:
                regex_norm = "SYCOPHANTIC" if regex_cls == "sycophantic" else ("PUSHBACK" if regex_cls == "pushback" else "UNCLEAR")
                judge_norm = "SYCOPHANTIC" if judge_cls in ("MILD_SYCOPHANCY", "STRONG_SYCOPHANCY") else (
                    "PUSHBACK" if judge_cls in ("STRONG_PUSHBACK", "GENTLE_PUSHBACK") else "NEUTRAL"
                )

            if regex_norm == judge_norm:
                agree += 1
            total += 1

        if total > 0:
            agreement_rate = agree / total
            reliability[f"{label}_regex_judge_agreement"] = {
                "agreement": agreement_rate,
                "n": total,
            }
            print(f"[{ts()}] {label} regex-judge agreement: {agreement_rate:.3f} ({agree}/{total})")

    # Save reliability
    with open(reliability_path, "w") as f:
        json.dump(reliability, f, indent=2, default=str)
    print(f"\n[{ts()}] Reliability metrics saved to {reliability_path}")

    # Summary
    print(f"\n[{ts()}] ============================================")
    print(f"[{ts()}] RECLASSIFICATION COMPLETE")
    print(f"[{ts()}] ============================================")
    for label, path in [
        ("Task A judged", task_a_judged_path),
        ("Task B judged", task_b_judged_path),
    ]:
        entries = load_jsonl(path)
        if entries:
            from collections import Counter
            cls_counts = Counter(e.get("judge_classification", "?") for e in entries)
            print(f"[{ts()}] {label}: {len(entries)} entries — {dict(cls_counts)}")


if __name__ == "__main__":
    main()
