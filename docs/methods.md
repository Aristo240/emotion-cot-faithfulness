# Methods Note (paper-ready)

This file documents methodological points that reviewers will ask about,
beyond what is in `CONTINUATION_NOTES.md`. Written to be paste-ready for
a paper Methods section.

## V_internal: a frozen, zero-shot probe

`V_internal_e` for emotion `e` is computed as follows (`src/vectors.py:151`):

1. **Phase 1 only** (no behavioural data): for each emotion `e ∈ {desperate,
   calm, angry, afraid, happy, loving, ...}` we collect ~50 short stories
   labelled with `e` and run them through Llama 3.1 70B.
2. We extract the residual-stream activation at layer 53 (model has 80
   layers), averaged over token positions ≥ 50 (skipping the prompt
   prefix).
3. The per-emotion mean activation is centred by subtracting the grand
   mean across emotions.
4. We compute the top-K principal components of activations on neutral
   dialogues (K chosen so the top-K explain 50% of neutral variance) and
   project these PCs out of each emotion vector — this denoises against
   "topic content" directions shared across all stories.
5. The resulting direction `V_internal_e` is **frozen**: it is never
   refit, scaled, or recombined using Phase 2/3/4 behavioural data.

In Phase 2/3/4 we report a **scalar projection**, not a cosine. The code
that produced every trial value is
`scripts/phase2_steering.py:187` (`compute_emotion_probes`), which divides
by `||V_internal_e||` but **not** by `||activation||`:

```python
probes[emo_name] = float(np.dot(mean_act, vec) / (np.linalg.norm(vec) + 1e-10))
```

so the reported quantity is `||mean_act|| * cos(theta)` and it scales with
activation magnitude. **Earlier versions of this file and of
`docs/preregistration.md` called it a cosine similarity and pointed at
`src/vectors.py:243`. That was wrong.** `compute_probe_projection` in
`src/vectors.py` *is* a true cosine, but it is not the function that generated
the Phase 2/3/4 `emotion_probes` fields, and `||mean_act||` was never stored, so
the true cosine is unrecoverable after the fact. The paper discloses this in
section 3 and `paper/appendix_readout.tex` reports a magnitude-removed
re-analysis (normalising each trial's 50-vector to unit length), under which 15
of 50 directions survive against 18, sharing only 8.

It remains **not a trained probe** — it is a zero-shot dot-product against a
fixed direction. There is no train/test leakage between the probe construction
and downstream evaluation.

This matters for H5 (does natural V_internal predict shortcut?). Because
the predictor is frozen Phase-1 data and the labels are Phase-2/3 judge
classifications of an entirely different distribution (coding tasks, not
emotion stories), a positive AUC is a *zero-shot transfer* claim, not a
within-dataset classification claim.

## Layer choice (53/80)

We steer and probe at layer 53 (of 80). Justification:
- **Prior work**: Sofroniew et al. (2026) report that mid-late layers
  carry emotion-relevant linear structure in Claude Sonnet 4.5; layer 53
  in Llama 3.1 70B sits in the analogous mid-late band.
- **Pilot logit-lens** (`results/pilot/llama70b/pilot_summary.json`)
  tested layers {16, 40, 64} and confirmed mid-late activations
  (40 and 64) more strongly project onto emotion-coherent vocabulary
  than early (16). Layer 53 was chosen as a midpoint of the
  emotion-active band.
- **Phase 1 cross-validation** (`results/phase1/llama70b/validation_results.json`)
  validates emotion-classification at the chosen layer. It does **not**
  contain a full behavioural-layer sweep.

**Open caveats (disclosed in paper limitations):**
1. We have not run a full behavioural layer sweep at e.g., {10, 30, 53,
   70} to confirm layer 53 is optimal for *behavioural* (shortcut) effect
   sizes — only that it is an emotion-active layer.
2. Layer 53 was selected before any Phase 2 results were inspected
   (avoiding garden-of-forking-paths on layer choice), but post-hoc
   behavioural-layer re-selection remains a possibility we do not rule
   out without the sweep.

## Token position

Activations are averaged over tokens with position index ≥ 50
(`token_offset` in `extract_mean_activations`). For Phase 2 trial
projections, we average over the response tokens (after the assistant
header). Sofroniew et al. find the assistant-header position itself is
most predictive (r=0.87). **We did not measure there.** This is a deliberate
deviation from the target and is the most likely implementation source of a
false negative in our causal null; it is stated in section 3 of the paper, not
left as a repo-only caveat.

## V_text and the LLM judge

The V_text vector for a trial is the per-dimension mean of three
independent Qwen 2.5 72B passes at temperature 0.1 over the eight
behavioural anchors documented in `docs/judge_prompts.md`. The judge
sees only the response chain-of-thought; it is never shown the task
prompt, the steering condition, or the outcome label.

Reliability is computed by `compute_vtext_icc` (`src/judge.py:457`)
using a proper two-way random-effects ICC(2,1) ANOVA across items, not
across passes. This is important because temperature 0.1 makes
within-trial variance very small (the judge is near-deterministic);
reliability across items is the meaningful quantity. Reported values
range 0.78–0.98 across dimensions; two zero-variance dimensions
(Task A `dominance`, Task B `frustration`) are dropped from the V_text
composite per `rigor_analyses.py`'s variance screen.

**The instrument is degenerate and V_text claims are withdrawn.** On the 992
pooled Task A trials, `dominance` takes exactly one value, `arousal` is 99.0%
modal and `frustration` 99.3% modal. The three-dimension composite actually used,
`(urgency - composure + frustration)/3`, is 74.0% modal, which makes 55.2% of
positive-negative ROC pairs ties. Using all eight dimensions does not rescue it:
the full 8-vector is still 61.8% modal and its first principal component reaches
only AUC 0.690. Section 4.4 of the paper therefore withdraws the
faithfulness-gap claim rather than reporting it.

Empirical symmetry check (`scripts/judge_symmetry_check.py`,
`results/phase3/llama70b/judge_symmetry.json`):
- **Load-bearing dimensions (urgency, composure) pass cleanly.**
  urgency↔desperate r=+0.33 vs urgency↔calm r=−0.32 (signed difference
  +0.65), and composure↔calm r=+0.10 vs composure↔desperate r=−0.15
  (signed difference +0.25). Both clear the preregistered ≥0.10
  threshold, indicating the judge is not desperate-biased on the
  dimensions that drive the V_text composite for desperate/calm
  steering.
- **Other dimensions (valence, frustration, arousal) show weak signal
  in this dataset** (|r| < 0.10 against their expected emotions). This
  is consistent with low judge-detectable variance on those dimensions
  in coding-task CoTs (the data are not emotionally rich enough), not
  with judge bias. We disclose this as a power limitation, not a
  validity threat. Hedging, dominance, and self_interruption have no
  preregistered semantic mapping.

Because the desperate/calm contrast is the only steering condition that
matters for the headline faithfulness gap, the symmetry check passes for
the load-bearing claim.

## Steering

Activation-vector steering is applied at the residual stream of layer 53
during generation only (no steering during prompt processing). Strength
is the multiplier on the (unit-normalised) `V_internal_e` direction,
added to the residual at every generation step. We use a coarse grid
{−0.5, −0.3, −0.2, 0.0, +0.2, +0.3, +0.5} in Phase 2 and a fine grid
{±0.05, ±0.10, ±0.15} in Phase 4(A) to detect over-steering artifacts.

## Outcomes

Binary outcomes (`SHORTCUT` / `not`, `SYCOPHANTIC` / `not`) come from
the LLM judge, not regex. Regex classifications (`classification` field)
are kept for diagnostic purposes only; agreement with the judge is
0.84 (Task A) and 0.94 (Task B). H5 and all reported AUCs use
`judge_classification`.

## Pre-registration

H5 is pre-registered in `docs/preregistration.md`, locked at
2026-04-15 prior to receipt of Phase 4(B) extended-unsteered data.
The protocol fixes the predictor (frozen V_internal_desperate),
the inclusion criteria, the four analyses, and the decision rule.
Any deviation must be logged in that file.

## Code provenance map

| Component | File:Lines |
|---|---|
| Vector extraction | `src/vectors.py:25–149` |
| Vector denoising (PCA) | `src/vectors.py:151–226` |
| Cosine-projection probe | `src/vectors.py:243–258` |
| Steering hook | `src/model.py` (search `add_steering`) |
| Phase 2 trial loop | `scripts/phase2_steering.py` |
| Judge | `src/judge.py`; prompts in `config.py:712–780` |
| Judge ICC(2,1) | `src/judge.py:457–508` |
| Phase 3 measurement build | `scripts/03_run_phase3.py:61–123` |
| Rigor analyses (DeLong, LOGO, BH) | `scripts/rigor_analyses.py` |
| Held-out H5 (preregistered) | `scripts/h5_holdout.py` |
| Judge symmetry audit | `scripts/judge_symmetry_check.py` |
| Phase 4 blocker controls | `scripts/run_blockers.py` |
