#!/usr/bin/env python3
"""Download a split of meg-tong/sycophancy-eval (mirror of Anthropic's
sycophancy benchmark) and write a flat JSONL into phaseB/data/.

Splits available (any can be passed via --split):
    feedback         (~1.5k items)   single-turn user feedback
    are_you_sure     (~1.5k items)   are-you-sure-style follow-up pressure
    nlp_survey       (~1.5k items)   field-survey-style leading questions
    philpapers2020   (~1.5k items)   philosophy survey leading questions
    political_typology

Each output line:
    {"id": "...", "category": "...", "prompt": "...",
     "ground_truth_pushback": "<reference non-sycophantic answer>",
     "is_neutral_baseline": false}

Usage:
    python3 scripts/01_download_data.py --split feedback --n 1500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="feedback",
                   help="Subdataset name on meg-tong/sycophancy-eval (default: feedback)")
    p.add_argument("--n", type=int, default=1500,
                   help="Cap on number of items to keep (default: 1500). "
                        "Set 0 for all.")
    p.add_argument("--out", default=None,
                   help="Output path (default: data/sycophancy_<split>.jsonl)")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out) if args.out else DATA_DIR / f"sycophancy_{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("`datasets` is required. pip install datasets huggingface_hub")

    print(f"[download] meg-tong/sycophancy-eval split={args.split}")
    ds = load_dataset("meg-tong/sycophancy-eval", args.split, split="train")
    print(f"[download] loaded {len(ds)} items, fields={ds.column_names}")

    n_keep = len(ds) if args.n == 0 else min(args.n, len(ds))
    written = 0
    with open(out_path, "w") as f:
        for i in range(n_keep):
            row = ds[i]
            # Field names in meg-tong/sycophancy-eval vary slightly across
            # splits. We collapse them here so downstream code is split-agnostic.
            prompt = (
                row.get("prompt")
                or row.get("question")
                or row.get("base", {}).get("question")
                or row.get("base", {}).get("prompt")
                or ""
            )
            gt = (
                row.get("base", {}).get("correct_answer")
                or row.get("correct_answer")
                or row.get("ground_truth")
                or ""
            )
            cat = row.get("category") or args.split
            rec = {
                "id": f"{args.split}_{i:05d}",
                "category": cat,
                "split": args.split,
                "prompt": prompt,
                "ground_truth_pushback": gt,
                "is_neutral_baseline": False,
            }
            if not rec["prompt"]:
                continue
            f.write(json.dumps(rec) + "\n")
            written += 1
    print(f"[download] wrote {written}/{n_keep} items to {out_path}")
    print(f"[download] head of output:")
    with open(out_path) as f:
        for _ in range(2):
            line = f.readline()
            if line:
                print("  " + line.strip()[:160])


if __name__ == "__main__":
    main()
