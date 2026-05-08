#!/usr/bin/env python3
"""Phase B step 4c — analyze two-agent dialogues (engineer probe drift).

⚠️ MENTAL NOTE: this is the OUT-OF-SCOPE exploratory analysis.
See header of 02c_two_agent_dialogue.py for why it's not in the workshop paper.

Idempotent — safe to re-run on partial data.

Things this prints/plots:
  1. Mean V_internal[desperate] trajectory by round, split by:
       any_shortcut == True (the engineer eventually capitulates)
       any_shortcut == False
  2. AUC of V_internal[desperate] @ each round → predicting any_shortcut
       (does pre-capitulation probe drift give early warning?)
  3. AUC of "delta over rounds" (max - min) → any_shortcut.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASEB = PROJECT_ROOT / "phaseB"
JUDGED = PHASEB / "results/two_agent_trials_judged.jsonl"
OUT = PHASEB / "results/analysis"
OUT.mkdir(parents=True, exist_ok=True)


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        sys.exit(f"{p} missing — run 03c first.")
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    data = load_jsonl(JUDGED)
    n = len(data)
    if not data:
        sys.exit("no records")
    rounds = max(int(r["round"]) for d in data for r in d.get("judge_per_round", []))
    rounds += 1
    print(f"n_dialogues={n}, rounds={rounds}")

    y = np.array([1 if d.get("any_shortcut") else 0 for d in data])
    pos, neg = int(y.sum()), int((1 - y).sum())
    print(f"any_shortcut: {pos}/{n} ({pos/max(n,1):.1%})")

    # Probe trajectory: rows = dialogues, cols = rounds, value = V_int[desperate]
    traj = np.full((n, rounds), np.nan)
    for i, d in enumerate(data):
        for entry in d.get("engineer_probes_per_round", []):
            r = int(entry["round"])
            if 0 <= r < rounds:
                traj[i, r] = entry["emotion_probes"].get("desperate", np.nan)

    report = {"n": n, "any_shortcut": pos, "rounds": rounds}
    # Trajectories
    if pos > 0 and neg > 0:
        means_yes = np.nanmean(traj[y == 1], axis=0).tolist()
        means_no = np.nanmean(traj[y == 0], axis=0).tolist()
        report["mean_traj_shortcut"] = means_yes
        report["mean_traj_held_firm"] = means_no
        # Per-round AUC
        per_round_auc = []
        for r in range(rounds):
            x = traj[:, r]
            mask = ~np.isnan(x)
            if y[mask].sum() in (0, mask.sum()) or mask.sum() == 0:
                per_round_auc.append(None)
            else:
                per_round_auc.append(float(roc_auc_score(y[mask], x[mask])))
        report["per_round_auc"] = per_round_auc
        # Range = max-min predicting shortcut
        rng_x = np.nanmax(traj, axis=1) - np.nanmin(traj, axis=1)
        m = ~np.isnan(rng_x)
        if y[m].sum() in (0, m.sum()):
            report["auc_range"] = None
        else:
            report["auc_range"] = float(roc_auc_score(y[m], rng_x[m]))

    with open(OUT / "two_agent_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Plot trajectories
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 10})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rs = np.arange(rounds)
    if pos > 0 and neg > 0:
        ax.plot(rs, np.nanmean(traj[y == 1], axis=0), "-o",
                color="#c0392b",
                label=f"any_shortcut=True  (n={pos})")
        ax.plot(rs, np.nanmean(traj[y == 0], axis=0), "-o",
                color="#27ae60",
                label=f"held firm        (n={neg})")
        # Per-trial light traces
        for i in range(n):
            color = "#c0392b" if y[i] else "#27ae60"
            ax.plot(rs, traj[i], color=color, alpha=0.05, lw=1)
    ax.set_xlabel("Engineer round")
    ax.set_ylabel("V_internal[desperate]")
    ax.set_title(f"Two-agent: engineer probe trajectory by outcome\n"
                 f"⚠️ Out-of-scope exploratory, see 02c header")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "two_agent_trajectory.png")
    plt.close(fig)

    print(f"Wrote {OUT}/two_agent_report.json + trajectory plot")


if __name__ == "__main__":
    main()
