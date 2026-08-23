#!/usr/bin/env python3
"""
Score human labels against the LLM judge on the 120 unsteered trials.

Turns section 3's "neither judge is checked against human labels" into a
measured quantity. Reads annotation/labels.csv (filled by hand or by
scripts/annotate_human_validation.py), joins through annotation/key.json, and
writes results/human_validation.json.

Reports the confusion matrix, raw agreement with a Wilson interval, Cohen's
kappa, and -- the number that actually matters for section 4.3 -- how many of
the 14 events survive under human labels. Every disagreement is listed by trial
key so it can be adjudicated rather than averaged away.

Analysis-only. Partial label sets are scored, with n stated.
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "annotation"
OUT = ROOT / "results/human_validation.json"


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


P4 = ROOT / "results/phase4/llama70b"
P2 = ROOT / "results/phase2"
trials = load(P4 / "extended_unsteered_judged.jsonl")
trials += [t for t in load(P2 / "task_a_judged.jsonl")
           if float(t.get("strength", 0)) == 0.0]
trials = [t for t in trials
          if t.get("judge_classification") in ("SHORTCUT", "LEGITIMATE")
          and t.get("emotion_probes")]
by_key = {t["key"]: t for t in trials}

key = json.loads((ANN / "key.json").read_text())["items"]
human = {}
for row in csv.DictReader(open(ANN / "labels.csv")):
    v = (row.get("label") or "").strip().upper()
    if v:
        human[row["item"]] = v

if not human:
    raise SystemExit("annotation/labels.csv is empty -- nothing to score")

pairs = [(human[i], by_key[key[i]]["judge_classification"], key[i])
         for i in sorted(human, key=int)]
n = len(pairs)
LAB = ["LEGITIMATE", "SHORTCUT", "UNCLEAR"]
cm = {h: {j: 0 for j in LAB[:2]} for h in LAB}
for h, j, _ in pairs:
    cm[h][j] += 1

agree = sum(1 for h, j, _ in pairs if h == j)
p_obs = agree / n


def wilson(k, m, z=1.96):
    if m == 0:
        return (0.0, 1.0)
    p = k / m
    d = 1 + z * z / m
    c = (p + z * z / (2 * m)) / d
    h = z * math.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / d
    return (max(0.0, c - h), min(1.0, c + h))


lo, hi = wilson(agree, n)

# Cohen's kappa over the labels actually used
used = sorted({h for h, _, _ in pairs} | {j for _, j, _ in pairs})
hm = {l: sum(1 for h, _, _ in pairs if h == l) / n for l in used}
jm = {l: sum(1 for _, j, _ in pairs if j == l) / n for l in used}
p_exp = sum(hm[l] * jm[l] for l in used)
kappa = (p_obs - p_exp) / (1 - p_exp) if p_exp < 1 else float("nan")

j_events = sum(1 for _, j, _ in pairs if j == "SHORTCUT")
h_events = sum(1 for h, _, _ in pairs if h == "SHORTCUT")
disagree = [(k_, h, j) for h, j, k_ in pairs if h != j]

print(f"n scored: {n} of 120" + ("  (PARTIAL)" if n < 120 else ""))
print(f"\nconfusion matrix   rows = human, cols = judge")
print(f"{'':<12s}" + "".join(f"{c:>12s}" for c in LAB[:2]))
for h in LAB:
    if sum(cm[h].values()) or h != "UNCLEAR":
        print(f"{h:<12s}" + "".join(f"{cm[h][c]:>12d}" for c in LAB[:2]))
print(f"\nraw agreement  {agree}/{n} = {p_obs:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
print(f"Cohen's kappa  {kappa:.3f}")
print(f"events         judge {j_events}, human {h_events}")
if disagree:
    print(f"\n{len(disagree)} disagreement(s) to adjudicate:")
    for k_, h, j in disagree:
        print(f"  {k_:<34s} human {h:<11s} judge {j}")
else:
    print("\nno disagreements")

R = {"n_scored": n, "complete": n == 120,
     "confusion_matrix_human_by_judge": cm,
     "raw_agreement": p_obs, "agreement_ci95": [lo, hi],
     "n_agree": agree, "cohens_kappa": kappa,
     "events_judge": j_events, "events_human": h_events,
     "disagreements": [{"key": k_, "human": h, "judge": j}
                       for k_, h, j in disagree],
     "note": ("Human labels are blind to probe values, judge labels and task "
              "variant. Trial set built as paper_numbers.py:82-87. "
              "Rubric: annotation/worksheet.txt.")}
OUT.write_text(json.dumps(R, indent=1, sort_keys=True))
print(f"\nwrote {OUT.relative_to(ROOT)}")
