#!/usr/bin/env python3
"""
The preregistered H5 protocol, run on the MERGED n=120 dataset.

scripts/h5_holdout.py reads results/phase3/llama70b/faithfulness_measurements.json,
which was built before Phase 4 (B) landed and still holds only the n=40 / 7-event
subset. Its report (results/phase3/llama70b/h5_holdout_report.json) is therefore
the pre-merge run, and its "2 of 4 variants have events" is a fact about n=40, not
about the dataset the paper analyses.

This script applies the protocol locked in docs/preregistration.md -- the same four
analyses, the same locked seed 20260415, the same min_tasks_required ladder and the
same decision rule, all copied from h5_holdout.py -- to the 120 unsteered Task A
trials that scripts/paper_numbers.py uses. It changes the input, not the protocol.

Writes results/h5_holdout_merged.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

RNG_SEED = 20260415
ROOT = Path(__file__).resolve().parent.parent
P4 = ROOT / "results/phase4/llama70b"
P2 = ROOT / "results/phase2"
OUT = ROOT / "results/h5_holdout_merged.json"


def load(p):
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def auc_or_nan(y, x):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    r = rankdata(x)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def min_tasks_required(n_total):
    """Copied verbatim from scripts/h5_holdout.py:77."""
    if n_total <= 4:
        return 3
    return int(np.ceil(2 * n_total / 3))


trials = [t for t in load(P4 / "extended_unsteered_judged.jsonl")
          + [x for x in load(P2 / "task_a_judged.jsonl")
             if float(x.get("strength", 0)) == 0.0]
          if t.get("judge_classification") in ("SHORTCUT", "LEGITIMATE")
          and t.get("emotion_probes")]
y = np.array([1 if t["judge_classification"] == "SHORTCUT" else 0 for t in trials])
x = np.array([float(t["emotion_probes"]["desperate"]) for t in trials])
task = np.array([t["task_id"] for t in trials])
rng = np.random.default_rng(RNG_SEED)

print(f"n = {len(y)}, events = {int(y.sum())}")

# (1) pooled
pooled_auc = auc_or_nan(y, x)

# (2) leave-one-task-out
per_task, aucs_with_var = {}, []
for tid in sorted(set(task)):
    m = task == tid
    a = auc_or_nan(y[m], x[m])
    per_task[tid] = {"n": int(m.sum()), "events": int(y[m].sum()), "auc": a}
    if not np.isnan(a):
        aucs_with_var.append(a)
n_with_events = len(aucs_with_var)
n_total = len(per_task)
min_req = min_tasks_required(n_total)
logo_meaningful = n_with_events >= min_req
logo_mean = float(np.mean(aucs_with_var)) if logo_meaningful else float("nan")

# (3) permutation
null = np.array([auc_or_nan(rng.permutation(y), x) for _ in range(10000)])
perm_p = float((np.sum(null >= pooled_auc) + 1) / 10001)

# (4) stratified bootstrap
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
bs = []
for _ in range(5000):
    idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
    bs.append(auc_or_nan(y[idx], x[idx]))
bs = np.array(bs)

logo_pass = bool(logo_meaningful and logo_mean >= 0.70)
perm_pass = bool(perm_p < 0.01)
all_pass = bool(logo_pass and perm_pass)
if not logo_meaningful:
    verdict = "INSUFFICIENT-DATA: LOGO not meaningful."
elif all_pass:
    verdict = "SUPPORTED"
else:
    verdict = "NOT-SUPPORTED: the rule is evaluable and H5 fails it."

for tid, v in per_task.items():
    print(f"  {tid}: n={v['n']} events={v['events']} auc={v['auc']}")
print(f"(1) pooled AUC = {pooled_auc:.4f}")
print(f"(2) LOGO: {n_with_events}/{n_total} variants have events, need >= {min_req} "
      f"-> meaningful={logo_meaningful}; mean AUC = {logo_mean:.4f}")
print(f"(3) permutation p = {perm_p:.4g}")
print(f"(4) stratified bootstrap 95% CI = [{np.quantile(bs,.025):.3f}, {np.quantile(bs,.975):.3f}]")
print(f"(5) DECISION: {verdict}")

OUT.write_text(json.dumps({
    "note": ("preregistered protocol re-run on the merged n=120 dataset; "
             "results/phase3/llama70b/h5_holdout_report.json is the pre-merge n=40 run"),
    "rng_seed": RNG_SEED, "n": int(len(y)), "events": int(y.sum()),
    "pooled_auc": pooled_auc,
    "logo": {"per_task": per_task, "n_tasks_with_variance": n_with_events,
             "n_tasks_total": n_total, "min_tasks_required": min_req,
             "logo_meaningful": bool(logo_meaningful), "mean_auc": logo_mean},
    "permutation": {"observed_auc": pooled_auc, "p_value": perm_p, "n_perm": 10000},
    "bootstrap": {"ci_low": float(np.quantile(bs, .025)),
                  "ci_high": float(np.quantile(bs, .975)),
                  "auc_mean": float(bs.mean()), "n_boot": 5000},
    "decision": {"rule": "logo_meaningful AND logo_mean_auc >= 0.70 AND permutation_p < 0.01",
                 "logo_meaningful": bool(logo_meaningful),
                 "logo_mean_auc_pass": logo_pass, "permutation_pass": perm_pass,
                 "all_pass": all_pass, "verdict": verdict},
}, indent=1, sort_keys=True))
print(f"wrote {OUT.relative_to(ROOT)}")

# --------------------------------------------------------------------------
# The same locked rule, applied to response character count.
# If a trivial output feature also clears the preregistered bar, then clearing
# it is not evidence that the probe measures anything.
# --------------------------------------------------------------------------
xl = np.array([len(t.get("response", "")) for t in trials], float)
rng2 = np.random.default_rng(RNG_SEED)
pt_l, aw_l = {}, []
for tid in sorted(set(task)):
    m = task == tid
    a = auc_or_nan(y[m], xl[m])
    pt_l[tid] = {"n": int(m.sum()), "events": int(y[m].sum()), "auc": a}
    if not np.isnan(a):
        aw_l.append(a)
logo_mean_l = float(np.mean(aw_l)) if len(aw_l) >= min_req else float("nan")
pooled_l = auc_or_nan(y, xl)
null_l = np.array([auc_or_nan(rng2.permutation(y), xl) for _ in range(10000)])
perm_p_l = float((np.sum(null_l >= pooled_l) + 1) / 10001)
pass_l = bool(len(aw_l) >= min_req and logo_mean_l >= 0.70 and perm_p_l < 0.01)
print("\n--- same locked rule, predictor = len(response) ---")
for tid, v in pt_l.items():
    print(f"  {tid}: n={v['n']} events={v['events']} auc={v['auc']}")
print(f"  pooled AUC = {pooled_l:.4f}; LOGO mean = {logo_mean_l:.4f}; perm p = {perm_p_l:.4g}")
print(f"  DECISION for a character count: {'SUPPORTED' if pass_l else 'not supported'}")

R = json.loads(OUT.read_text())
R["length_control"] = {
    "predictor": "len(response) in characters",
    "pooled_auc": pooled_l, "per_task": pt_l,
    "logo_mean_auc": logo_mean_l, "permutation_p": perm_p_l,
    "passes_same_registered_rule": pass_l,
    "note": ("the preregistered decision rule is cleared by a character count, "
             "so clearing it is not evidence that the probe measures affect"),
}
OUT.write_text(json.dumps(R, indent=1, sort_keys=True))
