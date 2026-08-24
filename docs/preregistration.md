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
   only the 4 original `fast_sum` variants remain, and 3 of 4 produce events
   (4/30, 0/30, 9/30, 1/30). **This entry as first written (2026-08-22) called
   that "below the scaled minimum" and recorded INSUFFICIENT-DATA. That was
   wrong; see deviation 5 below, logged 2026-08-24.**

4. **Analyses not in this registration.** The length control, the max-T sweep
   over all 50 directions, the conditional null, the nuisance-model
   robustness sweeps, the random-subspace control, the layer sweep and the
   V_text tie analysis are all **exploratory** and are labelled as such in the
   paper. Section 6 of this file anticipated that any such analysis would be
   reported as exploratory; that is what was done.

5. **Correction to deviation 3, logged 2026-08-24, after re-running the locked
   script.** Deviation 3 recorded INSUFFICIENT-DATA on the merged data. That
   was an error, and the corrected verdict is SUPPORTED. Both the threshold
   arithmetic and the resulting verdict are stated here in full, because this
   registration is the document a reader checks the paper's central claim
   against.

   *The threshold.* Section 5 states the rule over the 9-variant suite with a
   bar of >= 6. The implementation this registration names as binding,
   `scripts/h5_holdout.py`, does not hold the suite size fixed at 9: it reads
   `n_total` from the variants actually present and computes
   `min_tasks_required(n_total)`, which returns 3 for `n_total <= 4` and
   `ceil(2*n_total/3)` above that. For the 4 variants that remain, both routes
   give 3 (`ceil(8/3) = 3`), and the pre-Phase-4 run recorded in section 8
   already used that same minimum of 3. So 3 of 4 with events **meets** the
   threshold under the implementation; deviation 3's "below the scaled
   minimum" was an arithmetic error, not a different reading.

   *Both readings, stated.* There is a stricter reading on which the suite is
   the 9 registered variants and the bar is 6. Under it the rule is not
   evaluable at all, because the 5 variants that would have supplied the extra
   events are the ones deviation 2 excludes. The paper reports both readings
   explicitly (section 4.6) and claims a functional result under neither.

   *The verdict.* Re-running `scripts/h5_holdout_merged.py` on the merged
   n = 120 / events = 14 dataset returns, verbatim:
   `3/4 variants have events, need >= 3 -> meaningful=True`,
   LOTO mean AUC 0.7539 against the 0.70 bar, permutation p = 9.999e-05
   against the .01 bar, `DECISION: SUPPORTED`. Stored at
   `results/h5_holdout_merged.json`.

   *Why this is not a vindication.* The same locked rule, with
   `len(response)` in characters substituted for the probe, also returns
   SUPPORTED: LOTO mean 0.8798, permutation p = 9.999e-05, pooled AUC 0.8881,
   against the probe's 0.7539 and 0.8322. A criterion a character count clears
   does not license a claim about affect. That is the paper's finding, and it
   is why the corrected verdict strengthens no functional claim.

6. **The nested-model p in section 8 was computed against an invalid reference
   distribution, logged 2026-08-24.** Section 8 as first written reported
   `chi2(1) = 3.31, p = 0.069`. The statistic is a difference of unpenalized
   log-likelihoods evaluated at **ridge-penalized** estimates, for which the
   chi-square(1) reference distribution does not hold; the reported 0.069 is
   exactly `scipy.stats.chi2.sf(3.311045127775209, 1) = 0.0688157136359797`.
   The paper therefore never labels the statistic a chi-square and calls it
   `T`, taking all p-values from resampling instead. The per-direction
   conditional-null p for `desperate` is **0.060**
   (`results/conditional_null.json`, `desperate_marginal.p_conditional`
   = 0.060193980601939805) and the family-wise max-T p is **0.440**
   (`p_conditional.desperate`). Section 8 below is corrected accordingly.

## 8. Result (post-data)

To be filled in **after** Phase 4 (B) `extended_unsteered.jsonl` finishes
**AND** after the diverse Task A variants (`config.py: TASK_A_DIVERSE`)
have been collected, then `scripts/h5_holdout.py` re-run on the merged
dataset.

**Result (filled in 2026-08-22; corrected 2026-08-24).**
`scripts/h5_holdout.py` was re-run on the merged n=120 / events=14 dataset. The
diverse Task A variants were collected but are excluded for the judge-abstention
reason logged in section 7, so the suite at evaluation time is the 4 original
`fast_sum` variants, of which 3 produce outcome variance (4/30, 0/30, 9/30,
1/30).

> **The 2026-08-22 entry recorded this as "below the scaled minimum" and
> returned INSUFFICIENT-DATA. That was wrong.** `min_tasks_required(4)` returns
> 3, so 3 of 4 meets the threshold. The error and both available readings of
> the threshold are set out in deviation 5. What follows is the corrected
> result.

**Verdict: SUPPORTED.** Re-running `scripts/h5_holdout_merged.py` returns
`3/4 variants have events, need >= 3 -> meaningful=True`, LOTO mean AUC
**0.7539** against the 0.70 bar, permutation p = **9.999e-05** against the .01
bar, and `DECISION: SUPPORTED`. Stored at `results/h5_holdout_merged.json`.
Under the stricter reading, on which the suite is the full 9 registered
variants and the bar is 6, the rule is not evaluable at all, since the 5
variants that would have supplied the extra events are the ones deviation 2
excludes. The paper reports both readings (section 4.6).

**The corrected verdict licenses no functional claim, and the paper makes
none.** The same locked rule applied to `len(response)` in characters also
returns SUPPORTED — LOTO mean 0.8798, permutation p = 9.999e-05, pooled AUC
0.8881, against the probe's 0.7539 and 0.8322. A criterion that a character
count clears cannot distinguish a probe that measures affect from one that
measures how much text was produced. That is the paper's finding.

For completeness, the registered direction adds nothing detectable to a
length-only model: T = 3.31 at a per-direction conditional-null p of **0.060**,
and p = **0.440** under family-wise max-T correction across the 50 directions
(`results/conditional_null.json`). The 2026-08-22 entry gave this as
`chi2(1) = 3.31, p = 0.069`; that reference distribution does not apply to a
ridge-penalized statistic, and the correction is logged as deviation 6. These
numbers are reported in the paper as descriptive, not confirmatory.

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
