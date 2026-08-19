# Unfaithful by Feeling? Separating Semantic, Predictive, and Causal Validity for Emotion Probes in Llama 3.1 70B

> **Status (2026-08-19): this README supersedes the 2026-04-11 version.** Several
> headline numbers in the earlier README (`CV AUC 0.997`, `leave-one-task-out
> 0.992`, `steering p = 0.005`) **did not survive** the specificity and
> cross-validation controls run in April–May 2026. They are retracted below and
> the honest replacements are given. See [What changed and why](#what-changed-and-why).

## TL;DR

We extract linear emotion representations from Llama 3.1 70B's residual stream and
ask whether chain-of-thought text reveals the internal emotional state that
precedes reward-hacking behavior. Running four progressively stricter validity
checks, the claim **degrades in a specific and informative pattern**:

| # | Validity tier | Question | Verdict |
|---|---------------|----------|---------|
| 1 | **Semantic** | Do the probes track anything humans call emotion? | **Holds** (EmoBank valence CV R² = 0.38, n = 10,062) |
| 2 | **Behavioral association** | Does probe state co-vary with reward hacking with no intervention? | **Holds, concurrent and task-local** (AUC 0.832 [0.742, 0.909], n = 120, perm p < 0.001) |
| 3 | **Incremental over text** | Does it beat a CoT-text baseline on those same trials? | **Not at the pre-specified layer** (Δ = 0.037, p = 0.39) |
| 4 | **Causal specificity** | Does steering the emotion direction beat a random direction of matched norm? | **Fails** (Fisher p = 0.835) |
| — | **Preregistered generalization** | Does tier 2 hold across shortcut mechanisms? | **INSUFFICIENT-DATA** — criterion unmet, see below |

The short version: **a representation can be semantically valid and behaviorally
associated while providing no causal handle.** These properties need to be
validated separately; in this case study they dissociate.

> ⚠️ **Read this before quoting the tier-2 number.** `compute_emotion_probes`
> (`scripts/phase2_steering.py:187`) averages the residual stream from token 50 to
> the **end of the sequence**, which includes the generated response. The probe is
> therefore measured *concurrently with* the reward-hacking behavior, not before
> it. Tier 2 is an **association**, not prediction of a future choice, and part of
> it may reflect the shortcut text itself. A pre-decision readout at the
> assistant-header position exists in `src/experiments.py`
> (`compute_token_position_probes`) but **was not used** to generate any data
> analyzed here. Fixing this is the highest-value next experiment.

## Retracted claims

These appeared in the April README and should not be cited.

| Retracted claim | What actually holds | Evidence |
|---|---|---|
| "V_internal CV AUC **0.997**, leave-one-task-out **0.992**" | Pooled AUC is inflated by task-identity separability. Cross-mechanism transfer is **0.303** (below chance), perm p = 0.219 | `results/phase2/_lambda_partial/analysis_today/phase2_transfer_stats.json` |
| "Steering causally changes behavior, permutation **p = 0.005**" | Fine-grained judged sweep: desperate trend **z = −1.42, p = 0.157** (n.s.). Original p was uncorrected over 40+ strength × emotion tests | `results/phase4/llama70b/analysis_judged/report.json` → `trend_test` |
| "Desperate steering *decreases* shortcuts — novel model-specific finding" | Not distinguishable from a random direction. Emotion @0.3 = 7.5%, random = 6.5%, **Fisher p = 0.835** | same file → `random_null` |
| "Within-task CV AUC > 0.98 shows the signal is real" | True only *within* a task. Task-id residualization cuts AUC 0.799 → **0.674** | `results/phase3/llama70b/rigor_report.md` |
| H5 confirmed at AUC 0.955 / 0.901 | Preregistered decision rule returned **INSUFFICIENT-DATA** at the time (2/4 variants had outcome variance). H5 was only later supported on the *extended* set (below) | `results/phase3/llama70b/h5_holdout_report.json` |

## What holds

### 1. Semantic validity — probes track human emotion ratings

Zero-shot projection of the 50 Phase-1 emotion vectors onto EmoBank (10,062
human-rated sentences), probes frozen and never refit:

| Dimension | CV R² |
|---|---|
| Valence | **0.377** |
| Arousal | 0.180 |
| Dominance | 0.154 |

Valence is solidly recovered; arousal and dominance are weak. Top valence
correlates are `ecstatic` (r = +0.38), `excited` (+0.36), `furious` (−0.34),
`terrified` (−0.34) — the sign structure is coherent.
→ `results/emobank/llama70b/report.json`

### 2. Behavioral association — probe state co-varies with unsteered reward hacking

Unsteered trials only, no steering contamination. **Concurrent, not predictive**
(see the warning above), and **not a confirmatory H5 claim** (see below):

| | value |
|---|---|
| n / events | 120 / 14 |
| V_internal[desperate] AUC | **0.832** |
| bootstrap 95% CI | [0.742, 0.909] |
| permutation p | < 0.001 (10k) |
| Mann–Whitney p | 2.8 × 10⁻⁵ |
| V_internal[−calm] AUC | 0.865 |

**Judge-robust.** The n = 120 set is 80 trials from
`extended_unsteered_claude_judged.jsonl` plus 40 unsteered trials from
`task_a_claude_judged.jsonl`. Qwen 2.5 72B and Claude, judging independently,
agree on **120/120 trials (100%)** — 80/80 and 40/40 respectively — identifying
the same 14 shortcuts. The result is not an artifact of one judge.
→ `results/phase4/llama70b/analysis_judged/report.json` → `h5_extended`

**But the preregistered criterion is NOT met.** `docs/preregistration.md` §5 makes
the leave-one-task-out mean — not pooled AUC — the load-bearing metric, and
requires **≥6 of the 9 planned task variants** to produce outcome variance
(`min_tasks_required()` in `scripts/h5_holdout.py` scales 4→3, 9→6). The 5
diverse variants added to satisfy this did not yield usable labels (see
limitation 2), so the suite still has 2 variants with outcome variance. The
preregistered verdict remains **INSUFFICIENT-DATA**. A pooled AUC > 0.70 is
explicitly declared insufficient by the registration. **Do not cite tier 2 as a
confirmed H5 result.**

### 3. Incremental validity over CoT text — layer-dependent, partly exploratory

Pooled across steered and unsteered trials (n = 992, 87 events), the gap is large:

| Predictor | AUC | 95% CI |
|---|---|---|
| V_internal | 0.836 | [0.797, 0.872] |
| V_text (judge-rated CoT tone) | 0.647 | [0.603, 0.690] |
| **Δ** | **0.189** | **[0.141, 0.239]** |

The safety-relevant slice — trials where the probe is high but the text looks calm:

| Quadrant | n | hacks | rate |
|---|---|---|---|
| Both low | 493 | 4 | 0.8% |
| **HIDDEN** (V_int high, V_text low) | **453** | **66** | **14.6%** |
| Both high | 43 | 17 | 39.5% |

→ `results/phase2/_lambda_partial/analysis_today/faithfulness_gap_robust.json`

**But on the clean unsteered subset the gap is layer-dependent** (n = 80, 7 events):

| Layer | V_int[desp] AUC | Δ vs V_text | 95% CI | p (1-sided) |
|---|---|---|---|---|
| 13 | 0.691 | +0.053 | [−0.173, 0.299] | 0.339 |
| 26 | 0.450 | −0.187 | [−0.493, 0.125] | 0.890 |
| **39** | **0.918** | **+0.280** | **[0.056, 0.479]** | **0.008** |
| 52 | 0.708 | +0.069 | [−0.247, 0.343] | 0.306 |
| **53 (primary)** | 0.677 | +0.037 | [−0.282, 0.329] | 0.391 | 
| 65 | 0.472 | −0.165 | [−0.539, 0.202] | 0.809 |

**At the primary layer 53, the incremental gap over V_text is not significant on
unsteered trials.** Only layer 39 clears it, at p = 0.008 — and with 6 layers
tested, Bonferroni α = 0.0083, so it passes by a hair under Qwen labels
(p = 0.0080) and fails under Claude labels (p = 0.0097). **Treat layer 39 as
exploratory and in need of replication**, not as a confirmatory result.
→ `results/phase4/llama70b/analysis_layer_sweep/summary.json`

## What fails

### 4. Causal specificity — emotion steering is not distinguishable from noise

| Condition | hacks / n | rate |
|---|---|---|
| Unsteered baseline | 14 / 120 | 11.7% |
| Emotion vector @ ±0.3 | 12 / 160 | 7.5% |
| **5 random directions** (orthogonal to emotion subspace) @ ±0.3 | 13 / 200 | **6.5%** |

**Fisher emotion vs random: p = 0.835.** Steering along the emotion direction
does no more than steering along an arbitrary direction of the same norm. The
fine-grained dose–response over ±0.05…±0.5 shows no monotone trend
(desperate z = −1.42, p = 0.157; calm z = +2.04, p = 0.041 uncorrected, wrong
sign, does not survive correction).

Text injection ("feel desperate" in the system prompt) also fails to separate:
17.5% vs 11.7% baseline, Fisher p = 0.417.

**This is the paper's central negative result** and the reason the causal framing
of the April README is withdrawn.

## Known limitations — read before reusing any number

0. **The probe is a response-mean readout, so tier 2 is concurrent.** See the
   warning at the top. `TOKEN_OFFSET = 50` to end-of-sequence, spanning the
   generated response. No result here licenses a claim about detecting an
   internal state *before* the model commits to a shortcut.
1. **V_text is a low-resolution instrument.** Its distribution is 946 low / **0
   mid** / 46 high with only 14 unique rounded values (n = 992). Part of
   "V_internal beats V_text" is "V_text barely varies." Do not read the gap as a
   clean statement about CoT faithfulness until V_text is remeasured on a scale
   with real variance.
2. **The diverse-mechanism dataset is not usable as-is.** On
   `task_a_diverse_claude_judged.jsonl` (n = 650), the two judges agree on only
   **165/650 trials (25.4%)**: Qwen labels 2 SHORTCUT / 524 UNCLEAR, Claude
   labels 194 SHORTCUT / 39 UNCLEAR. **Any transfer/generalization conclusion
   from this file — including the negative AUC 0.303 — is unreliable** and
   blocked on judge reconciliation.
3. **Probe–task confound.** Probes predict task identity at 95.4% accuracy
   (chance 25%); task_id explains 68–1800× more probe variance than emotion.
   Residualizing on task_id cuts AUC 0.799 → 0.674.
4. **Underpowered throughout.** 14 events at n = 120 for the headline result.
5. **Task B (sycophancy) is dead.** 2 sycophantic responses in 650 trials
   (0.31%). Excluded from all conclusions.
6. **Layer 39 is a post-hoc selection** from a 6-layer sweep. See above.
7. **`phaseB/` has scripts but no results.** It has never been run.

## Provenance table

Every claim above, traced to the script and file that produced it.

| Claim | Script | Input | Result file | Status |
|---|---|---|---|---|
| EmoBank valence R² = 0.377 | `scripts/run_emobank_validation.py` | EmoBank (10,062 sents) | `results/emobank/llama70b/report.json` | Confirmatory |
| Association AUC = 0.832 (n=120) | `scripts/analyze_phase4_judged.py` | `extended_unsteered_judged.jsonl` + unsteered rows of `task_a_judged.jsonl` | `results/phase4/llama70b/analysis_judged/report.json` → `h5_extended` | **Exploratory** — prereg criterion unmet |
| Qwen/Claude agree 120/120 | — (direct file comparison) | `extended_unsteered_claude_judged.jsonl` (80) + `task_a_claude_judged.jsonl` @ strength 0 (40) | — | Confirmatory |
| Pooled gap Δ = 0.189 | `scripts/analyze_faithfulness_gap_robust.py` | fast_sum family, `judge_classification` | `.../analysis_today/faithfulness_gap_robust.json` | Exploratory (pooled; task confound) |
| Layer sweep, layer 39 | `scripts/analyze_layer_sweep.py` | `extended_unsteered_layer_sweep.jsonl` | `.../analysis_layer_sweep/summary.json` | **Exploratory** (6-layer selection) |
| Random-direction null, p = 0.835 | `scripts/analyze_phase4_judged.py` | `random_directions_judged.jsonl` | `.../analysis_judged/report.json` → `random_null` | Confirmatory (preplanned control) |
| Dose–response n.s. | `scripts/analyze_phase4_judged.py` | `finegrained_judged.jsonl` | `.../analysis_judged/report.json` → `trend_test` | Confirmatory (preplanned control) |
| Transfer AUC = 0.303 | `scripts/analyze_phase2_transfer_stats.py` | `task_a_diverse_claude_judged.jsonl` | `.../analysis_today/phase2_transfer_stats.json` | **Unreliable** — see limitation 2 |
| H5 INSUFFICIENT-DATA verdict | `scripts/h5_holdout.py` | `faithfulness_measurements.json` | `results/phase3/llama70b/h5_holdout_report.json` | Confirmatory (preregistered) |

### Label field semantics — important

Judged `.jsonl` rows carry **three** classification fields. Using the wrong one
silently changes every downstream number:

| Field | Meaning |
|---|---|
| `classification` | **Stale regex heuristic.** In `task_a_diverse_*` it is `unclear` for all 650 rows. **Never use for analysis.** |
| `judge_classification` | Qwen 2.5 72B, 3-pass majority |
| `claude_classification` | Claude, 3-pass majority (cross-family check) |

Analysis scripts are not uniform: `analyze_phase4_judged.py` and
`analyze_faithfulness_gap_robust.py` read `judge_classification`, while
`analyze_diverse_today.py` and `analyze_phase2_transfer_stats.py` read
`claude_classification`. Check the field before comparing numbers across scripts.

## What changed and why

The April README reported the first-pass analysis. Between 2026-04-14 and
2026-05-12 we ran the controls that first-pass analysis lacked:

1. **Multiple-comparison correction** (`scripts/rigor_analyses.py`, BH-FDR) —
   removed the per-strength steering effects.
2. **Random-direction and text-injection specificity controls**
   (`scripts/run_blockers.py`) — removed the causal claim entirely.
3. **A preregistered H5 protocol with a binding decision rule**
   (`docs/preregistration.md`, RNG seed 20260415) — which returned
   INSUFFICIENT-DATA on the original data and was only satisfied after
   collecting 80 additional unsteered trials.
4. **A cross-family judge** (Claude alongside Qwen) — which validated the
   headline result and invalidated the diverse-mechanism dataset.

We report this trajectory rather than only the endpoint, because the pattern of
*which* claims died under *which* control is the substantive finding.

## Methodology

```
Phase 1  50 emotion vectors from Llama 3.1 70B residual stream (primary layer 53/80)
   |     activation extraction + PCA denoising
   v
Phase 2  Steering with desperate/calm at 7 strengths (-0.5 … +0.5)
   |     Task A: impossible coding tasks (reward hacking)   [Task B dropped: 2/650 events]
   v
Phase 4  CONTROLS: fine-grained sweep (±0.05…±0.5), 5 random orthogonal directions,
   |     text injection, 80 extra unsteered trials, 6-layer probe sweep
   v
Judges   Qwen 2.5 72B (3-pass) + Claude (3-pass, cross-family)
   v
Analysis Preregistered H5 decision rule, BH-FDR, bootstrap CIs, permutation tests
```

- **V_internal:** cosine similarity between residual-stream activations and
  frozen Phase-1 emotion vectors. Never refit on behavioral data — H5 is a
  transfer claim, not a within-dataset classification claim (`docs/methods.md`).
- **V_text:** 8-dimension emotional tone rating (1–7) by the LLM judge.
  Prompts reproduced in `docs/judge_prompts.md`.

## Setup

```bash
conda create -n emotion-cot python=3.11 -y
conda activate emotion-cot
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia -y   # NOT cu130: host driver is 550.120
pip install -r requirements.txt
```

**Hardware:** 8× V100 32GB (float16, pipeline parallel), or 4× A100 80GB. ~500GB storage.

**Models:** `meta-llama/Llama-3.1-70B-Instruct` (steered), `Qwen/Qwen2.5-72B-Instruct` (judge).

## Reproducing the surviving results

```bash
# Confirmatory: natural probe state predicts unsteered reward hacking (+ all controls)
python scripts/analyze_phase4_judged.py

# Semantic validity against human ratings
python scripts/run_emobank_validation.py

# Exploratory: layer sweep
python scripts/analyze_layer_sweep.py

# Preregistered H5 decision rule
python scripts/h5_holdout.py
```

## Open questions

1. Does the layer-39 gap replicate on an independently collected unsteered set?
2. Can V_text be remeasured on a scale with genuine variance?
3. Does judge disagreement on the diverse set reflect genuine task ambiguity or a
   prompt failure for Qwen?
4. Is the absent causal effect specific to Llama, or did prior positive results
   lack random-direction controls?

## References

```bibtex
@article{sofroniew2026emotion,
  title={Emotion Concepts and their Function in a Large Language Model},
  author={Sofroniew, Nicholas and Kauvar, Isaac and others},
  journal={Transformer Circuits Thread},
  year={2026}
}
```
