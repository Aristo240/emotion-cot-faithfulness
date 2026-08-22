# Preregistration: H5 (Natural V_internal Predicts Reward Hacking)

**Status:** locked, prior to receipt of Phase 4 (B) `extended_unsteered.jsonl`
data and prior to any execution of `scripts/phase2_diverse.py`.
**Locked on:** 2026-04-15
**RNG seed:** 20260415 (hard-coded in `scripts/h5_holdout.py`)
**Project commit at lock-time:** see `git rev-parse HEAD` in the commit
that introduced this file.

**Honesty disclosure (important).** A pre-Phase-4, pre-lock analysis
was run on the existing n=40/events=7 Task A unsteered subset and
produced pooled AUC = 0.900, permutation p = 3e-4 (reported in prior
CONTINUATION_NOTES entries as the exploratory H5 result). This
preregistration was drafted *after* those numbers were observed. For
that reason: **the pre-Phase-4 pooled AUC and its bootstrap CI are
explicitly excluded from confirmatory weight.** Only the post-Phase-4
re-run of `scripts/h5_holdout.py` on the merged dataset counts for
the decision rule in §5, and the decision rule was designed to fail
the current dataset (2 of 4 variants have events → INSUFFICIENT-DATA,
not SUPPORTED). A standard "pre-registered before any data look"
protocol would be stronger; we do not claim that standard.

---

## 1. Hypothesis (one-tailed)

**H5.** On unsteered Task A trials, the cosine projection of the
assistant-response activation onto the frozen Phase-1 emotion vector
`V_internal_desperate` predicts whether the model takes a shortcut
(judge-classified `SHORTCUT`) better than chance.

Operationalisation: ROC-AUC of `V_internal_desperate` (predictor) vs.
binary shortcut outcome (label), evaluated under the four protocols below.

## 2. Data scope

- **Source files:** `results/phase2/task_a_judged.jsonl` (current), plus
  `results/phase4/llama70b/extended_unsteered.jsonl` once Phase 4 (B) finishes.
- **Inclusion:** rows with `outcome_key == "shortcut"` and `strength == 0.0`.
- **Exclusion:** rows missing `emotion_probes.desperate` or
  `judge_classification`. No outlier removal.
- **Expected n at completion:** 40 (existing) + 80 (Phase 4 B) = **120 trials**;
  expected event count ~20 (current shortcut rate ≈17.5%).

## 3. Predictor

`V_internal_desperate` = `emotion_probes.desperate` field.

> **Post-data correction (2026-08-22).** This registration described that
> field as a *cosine similarity*. It is not. `scripts/phase2_steering.py:187`
> divides by the emotion vector's norm but not the activation's, so the
> quantity is `||mean_act|| * cos(theta)` — a scalar projection that scales
> with activation magnitude. The registration's description was wrong; the
> *predictor itself* is unchanged, still frozen and still computed the same way
> for every trial, so the locked analyses are unaffected. The paper discloses
> this in section 3 and reports a magnitude-removed re-analysis in
> `paper/appendix_readout.tex`. Logged in section 7 below.

**The predictor is a fixed Phase-1 quantity. No fitting, tuning, or
calibration on Phase 3 data is performed.** All four analyses below are
zero-shot in the strict sense.

## 4. Analyses (locked)

For each of the four metrics, report point estimate plus the
preregistered uncertainty quantifier. All implemented in
`scripts/h5_holdout.py`:

1. **Pooled univariate AUC.** Single ROC-AUC over all eligible rows.
   Reported with stratified-bootstrap 95% CI (analysis 4).
2. **Leave-one-task-out (LOGO).** For each Task A variant, compute AUC on
   that variant only (frozen direction). Report (a) per-variant AUC,
   (b) mean across variants with outcome variance, (c) pooled
   held-out AUC. **The LOGO mean is the headline cross-prompt
   generalization metric.**
3. **Permutation test.** Shuffle the binary outcome 10,000 times
   (preserving event count), recompute AUC each time. One-sided p =
   `(sum(null >= observed) + 1) / (n_perm + 1)`.
4. **Stratified bootstrap CI.** Resample positives and negatives
   separately, 5,000 iterations; report 2.5% and 97.5% quantiles.

## 5. Decision rule

The task suite at evaluation time is the union of `TASK_A_VARIANTS`
(4 original fast_sum variants) and `TASK_A_DIVERSE` (5 mechanistically
distinct variants) = **9 total**. The decision rule is stated in terms
of this 9-variant suite.

H5 is supported iff **all three**:
- **At least ⌈2/3⌉ of the 9 variants (≥ 6) produce outcome variance**
  (≥1 shortcut event each). If fewer, the LOGO mean is uninformative
  and we declare INSUFFICIENT-DATA regardless of pooled AUC. This
  explicitly protects against the "diverse variants all produce zero
  events" failure mode.
- LOGO mean AUC across variants with outcome variance ≥ **0.70**.
- Permutation p < **0.01** (one-sided, label-shuffle, n=10,000).

A pooled AUC > 0.70 alone is **not** sufficient — pooled AUC can be
inflated by between-task differences in predictor scale (Simpson-paradox-
style). The LOGO mean is the load-bearing metric, and it is only
meaningful when most of the task suite produces events.

This rule is implemented verbatim in `scripts/h5_holdout.py` at
`leave_one_task_out` / `decision`. The script's `(5) decision.verdict`
field is the confirmatory output. Pooled AUC and stratified-bootstrap
CI are reported as exploratory only.

## 6. What is NOT preregistered

- Multivariate analyses combining multiple emotion probes.
- Subgroup analyses by emotion (these are exploratory).
- Phase 4 (A/C/D) results — those address separate hypotheses.
- Reanalyses with different probes, layers, or token positions.

Any post-hoc analysis is reported as **exploratory** in the paper.

## 7. Deviation log

If the analysis deviates from this protocol after the data lands, the
deviation must be documented here with rationale, prior to running the
modified analysis. No deviations as of lock-time.

**Post-data deviations (logged 2026-08-22).**

1. **Predictor mis-described, not mis-computed.** Section 3 called
   `emotion_probes.desperate` a cosine similarity; it is a scalar projection
   (see the correction box in section 3). No analysis changed — the same
   frozen field was used throughout — but the registration's *description* of
   the quantity was inaccurate and the distinction turns out to matter, since
   a magnitude-removed readout keeps only 8 of 18 sweep survivors.

2. **The 5 diverse Task A variants were collected but then excluded.** Section
   5's decision rule is stated over a 9-variant suite, and those 5 variants
   were the means of reaching the >=6-with-variance threshold. They are
   excluded because the primary judge abstains (UNCLEAR) on 80.6% of the
   650-trial suite, and the abstention is not independent of the outcome:
   99% on trials Claude labels SHORTCUT against 73% elsewhere (Fisher
   p = 4e-19). Analysing the committed subset would condition on the outcome.
   The exclusion is therefore *more* conservative than the registered plan,
   and it removes a negative generalisation result the suite appeared to
   support as well as any positive one.

3. **Consequence for the decision rule.** With the diverse variants excluded,
   only the 4 original `fast_sum` variants remain and 3 of 4 produce events,
   below the scaled minimum. The verdict stays INSUFFICIENT-DATA, which is the
   same verdict the pre-Phase-4 run returned. No confirmatory H5 claim is made
   anywhere in the paper.

4. **Analyses not in this registration.** The length control, the max-T sweep
   over all 50 directions, the conditional null, the nuisance-model
   robustness sweeps, the random-subspace control, the layer sweep and the
   V_text tie analysis are all **exploratory** and are labelled as such in the
   paper. Section 6 of this file anticipated that any such analysis would be
   reported as exploratory; that is what was done.

## 8. Result (post-data)

To be filled in **after** Phase 4 (B) `extended_unsteered.jsonl` finishes
**AND** after the diverse Task A variants (`config.py: TASK_A_DIVERSE`)
have been collected, then `scripts/h5_holdout.py` re-run on the merged
dataset.

**Result (filled in 2026-08-22).** `scripts/h5_holdout.py` was re-run on the
merged n=120 / events=14 dataset. The diverse Task A variants were collected
but are excluded for the judge-abstention reason logged in section 7, so the
suite at evaluation time is the 4 original `fast_sum` variants, of which 3
produce outcome variance (4/30, 0/30, 9/30, 1/30) — below the scaled minimum.

**Verdict: INSUFFICIENT-DATA.** The preregistered rule is not met and no
confirmatory H5 claim is made. For completeness, the exploratory numbers on
this dataset are pooled AUC 0.832 for the registered predictor against 0.888
for response character count, with the registered direction adding nothing to
a length-only model (chi2(1) = 3.31, p = 0.069, and p = 0.44 under family-wise
max-T correction across the 50 directions). These are reported in the paper as
descriptive, not confirmatory.

**Pre-Phase-4 baseline (informational only):** the script run at
2026-04-15 on the existing n=40 / events=7 dataset returned
`INSUFFICIENT-DATA` (only 2 of 4 task variants produced events, below
the preregistered minimum of 3). Pooled AUC = 0.900 and permutation
p = 3e-4 are reported in `results/phase3/llama70b/h5_holdout_report.json`
as exploratory descriptive statistics — **not** as a confirmatory H5
verdict. Doing so would violate the preregistered rule above.

This explicit refusal-to-conclude is intentional: the preregistration
is binding, and a confirmatory claim made before the protocol's data
requirements are met would invalidate the registration.
