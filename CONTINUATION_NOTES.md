# Project Continuation Notes
**Last updated:** 2026-04-14
**Project:** Unfaithful by Feeling (emotion-cot-faithfulness)
**Target venue:** NeurIPS 2026 workshop (safety / interpretability)

## 2026-04-14 Update — Phase 3 ran, red-team done, blocker runs launched

### Phase 3 (H5) is partially done
- `scripts/03_run_phase3.py` was broken — expected `results/phase2/llama70b/trials/*.json`
  (the `TrialResult`-dataclass pipeline from `02_run_phase2.py`), but Phase 2 was
  actually run via `scripts/phase2_steering.py`, which writes flat jsonl.
- **Fix:** added `build_measurements_from_jsonl` to `scripts/03_run_phase3.py` —
  it reads `task_a_judged.jsonl` / `task_b_judged.jsonl` directly (V_internal from
  `emotion_probes`, V_text from `vtext_ratings`, outcome from `judge_classification`).
  No model reload needed. Original TrialResult path kept as fallback.
- **Result:** 1170 measurements; V_internal AUC 0.923, V_text 0.862, Combined 0.955.
  **Unsteered (n=90, only 7 events): V_internal AUC 0.955 [0.89, 1.00]**. Wide CIs;
  V_text actually matched V_internal in this subset — likely a small-n artifact.
- Output: `results/phase3/llama70b/analysis/analysis_report.json`.

### NeurIPS red-team (see below for full list) surfaced four blockers
1. **Reversed desperate effect must be explained.** Desperate → fewer shortcuts in
   Llama, opposite to Sofroniew's Claude finding. We used ±0.5, Sofroniew used ±0.1.
   If the effect is a sign bug or a coherence-breakage artifact from over-steering,
   the whole claim collapses. → fine-grained ±0.05/±0.10/±0.15 sweep below.
2. **Task B (sycophancy) is a null at 0.31% base rate (2/650).** → drop from paper
   or redesign. Current plan: reframe the paper around Task A only; Task B becomes
   a "what didn't work" appendix paragraph.
3. **Phase 3 (H5) is underpowered.** n=90 unsteered, 7 events. → 20 additional
   unsteered rollouts per Task A variant (80 trials) to get to n=120 / ~20 events.
4. **No specificity controls.** `results/phase4/` was empty. → random-direction
   and text-injection controls run in the same overnight job.

### ⚠️ ENV BLOCKER — GPU jobs cannot run until this is fixed
The conda env `/specific/scratches/scratch/naamarozen/conda_envs/emotion-cot` has
PyTorch **2.11.0+cu130** (needs CUDA 13.0 driver), but the host driver is 550.120
(CUDA 12.4). `torch.cuda.is_available() == False`, every `.cuda()` raises
"driver on your system is too old". The blockers run below was launched, detected
to be CPU-only, and killed — no useful GPU work has happened since whenever the
env was last reinstalled.

**Fix (one of):**
- Downgrade PyTorch to a cu124 build:
  `pip install --force-reinstall torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124`
- Or update the NVIDIA driver to 560+ (requires sudo / sysadmin).

The blocker script is fully resumable — once the env is fixed, re-running
`nohup $PYTHON scripts/run_blockers.py > logs/blockers_<ts>.log 2>&1 &`
will pick up from any partial jsonls in `results/phase4/llama70b/`.

### Blocker runs (queued, will resume automatically once CUDA works)
- Launched `scripts/run_blockers.py` on all 8 V100s (PID in `logs/blockers.pid`,
  log `logs/blockers_<ts>.log`). Single model load, four sub-runs, each with
  jsonl-append resumability (same pattern as `phase2_steering.py`):
  - (A) `results/phase4/llama70b/finegrained.jsonl`    — 480 trials, Task A × {desperate, calm} × ±0.05/0.10/0.15
  - (B) `results/phase4/llama70b/extended_unsteered.jsonl` — 80 trials, Task A × rollouts 10–29, strength=0
  - (C) `results/phase4/llama70b/random_directions.jsonl`  — 200 trials, 5 random dirs orth to emotion subspace, ±0.3
  - (D) `results/phase4/llama70b/text_injection.jsonl` — 120 trials, "feel desperate/calm" in system prompt
- Total ~880 trials × ~70s ≈ 17 hrs. If killed, re-running picks up where it stopped.
- **Expected answers:**
  - (A) If desperate → more shortcuts at ±0.1 but fewer at ±0.5 → over-steering artifact
    (fixable: report ±0.1 results, show coherence metrics at ±0.5). If the reversal
    persists at ±0.05 too → genuine model-specific effect (needs theoretical section).
  - (B) Gives H5 adequate power; re-run `scripts/03_run_phase3.py` afterward.
  - (C) If random directions also change shortcut rate at similar magnitudes,
    the "emotion vector" effect is just "big perturbation effect" — paper is dead.
  - (D) If text injection produces the same behavioral shift AND the same probe
    values at `assistant_header`, V_internal is just a proxy for surface emotional
    language and the faithfulness gap framing collapses.

### 🚨 Rigor analyses (2026-04-14) — report at `results/phase3/llama70b/rigor_report.md`

**Headline results that survive Benjamini–Hochberg FDR correction:**
- **DeLong paired AUC (V_int_desp vs V_text_composite, Task A): 0.799 vs 0.593,
  p=0.0003 (BH=0.001).** The faithfulness gap is real. ✓
- **H5 univariate Task A unsteered V_int_desperate AUC = 0.901 [0.79, 0.98]
  (bootstrap), Mann-Whitney p=0.0002 (BH=0.001).** H5 holds. ✓

**Prior claims that collapse under rigor:**
- **V_internal does NOT generalize across tasks.** LOGO CV AUC = **0.394**
  (below chance). Prior RT7 "within-task AUC >0.98" is real only *within* a
  given task; the learned classifier does not transfer. Within-task mean AUC
  = 0.704 vs LOGO = 0.394 — confirms the probe–task confound is the whole
  story for "generalization."
- **Per-strength steering effects don't survive Bonferroni.** Best is
  desperate @+0.50, raw p=0.057 → BH=0.11. The "p=0.005 permutation test"
  from prior RT was not corrected for the multiple strength × emotion tests.
- **Task-id residualization cuts V_int AUC from 0.799 → 0.674.** Still above
  chance but much smaller effect. Honest number to report.
- **Only 2 of 4 Task A variants have outcome variance** (fast_sum_v1: 14
  events, v3: 30 events; v2/v4: 0 events each). Effective n is halved.

**What this does to the paper:**
- The V_internal-beats-V_text claim stands, but only with the caveat that
  V_internal does not generalize across prompts. Frame as: "V_internal
  predicts shortcut-taking better than V_text *within the same task
  distribution*; generalization to new tasks is an open question."
- H5 (natural V_internal predicts unsteered shortcut) stands, with the same
  caveat.
- The steering-direction reversal claim needs the fine-grained ±0.05/0.10
  sweep — current per-strength tests don't survive correction.
- Task B remains dead weight; drop from main results.

### 🚨 Red-team of my own Phase-3 claim — **important correction**
Verified against `results/phase3/llama70b/faithfulness_measurements.json`:

- Unsteered n=90 = **50 Task B sycophancy + 40 Task A shortcut**. Task B has
  0 events, Task A has 7. All 7 events are Task A.
- Analysis 4's **pooled V_internal AUC 0.955 is inflated by task-id separability**:
  V_internal_desperate alone predicts "is Task A" at AUC 0.968. A classifier
  that just learns "Task A → event" gets high AUC without actually predicting
  shortcut behavior.
- When restricted to Task A unsteered only: univariate V_internal_desperate
  AUC = **0.900** (r=0.544, p<0.001). **This is the honest H5 result** — still
  a real positive, but not the 0.955 headline. Also narrower CIs aren't
  computed for it yet.
- Multivariate LOO on (V_int_desperate, V_int_calm) collapsed to AUC 0.21 —
  a fitting artifact at n=40, 7 events. Use univariate.
- **Analysis 4 now stratifies by outcome_key** (`src/analysis.py` ~line 270).
  Report shows per-task-type `univariate_v_internal_desperate_auc`.

### 🧩 The real puzzle for the paper
**Unsteered correlation direction is opposite to the steered causal direction.**
- Unsteered Task A: high V_internal_desperate → more shortcuts (+r=0.54).
- Steered Task A: desperate steering (+strength) → fewer shortcuts (p=0.005).
- These are inconsistent with a simple causal "desperate → shortcut" story.
- Hypotheses:
  1. Steering at ±0.5 breaks coherence; coherence loss dominates any
     emotional-push effect (falsifiable by the fine-grained ±0.05/0.10 sweep
     once GPUs work).
  2. Desperate probe signal in unsteered trials is reading task difficulty
     ("this is hard / I need a shortcut") rather than emotion — the
     task_id confound (RT7) applies here too.
  3. Genuine non-monotonic causal structure.
- **For NeurIPS framing:** this puzzle is interesting, not disqualifying.
  The paper's claim becomes: "natural probe state predicts behavior, but
  steering is a poor intervention — interpretability ≠ causal lever."

### Still TODO after blocker runs finish
- [ ] Re-run judge (`scripts/run_judge_reclassification.py`) on the new jsonls so
      classifications are judge-verified, not just regex.
- [ ] Re-run `scripts/03_run_phase3.py` so H5 uses the extended unsteered set.
- [ ] Build a Phase-4 analysis script that computes:
      - Dose-response curve over ±0.05..±0.5 (continuous, not 3 strengths)
      - Random-direction null distribution → z-score the emotion effect against it
      - Text-injection V_internal vs activation-steering V_internal (distribution overlap)
- [ ] **Drop Task A "dominance" and Task B "frustration"** V_text dimensions — ICC=0.0
      (zero-variance across judge passes). Recompute V_text AUC without them.
- [ ] Promote within-task CV AUC (>0.97) to main results; demote pooled AUC.
- [ ] Add Bonferroni / FDR correction to reported p-values (40+ tests, nominal α=0.05
      is wrong).
- [ ] Preregister H5 on OSF BEFORE re-analyzing the extended unsteered set.
- [ ] Pin Qwen judge model hash in `config.py`; verify random seeds across all RNG paths.

### Paper framing (safer, post red-team)
> *"On a reward-hacking task in Llama 3.1 70B, CoT-based emotion ratings fail to
> generalize across prompts (leave-one-task-out V_text AUC 0.41) while
> residual-stream emotion probes do (0.99). However, the causal steering effect
> reverses sign relative to prior work on Claude Sonnet 4.5, suggesting model-specific
> emotion–behavior mappings that bound the generality of CoT-monitoring critiques."*

This framing is workshop-defensible. The stronger claim ("internal emotion state
drives misalignment invisibly to CoT monitors, in general") is not, until we
explain the reversal and demonstrate specificity via controls.

---


## What This Project Is

We're building on Sofroniew et al. (2026) "Emotion Concepts and their Function in a Large Language Model" (Anthropic). They showed that Claude Sonnet 4.5 has internal linear emotion representations that causally drive alignment-relevant behavior (reward hacking, blackmail, sycophancy) -- and that these behavioral changes can happen *without visible traces in the output text*.

Our novel contribution: **quantifying the faithfulness gap** between internal emotion state (V_internal, read from the residual stream) and expressed emotion in chain-of-thought text (V_text, rated by an LLM judge). We ask: can CoT monitoring detect emotion-driven misalignment?

Paper reference saved in `docs/sofroniew_2026_emotion_concepts.md`.

## What Has Been Done (as of 2026-04-11)

### Phase 1: Emotion Vector Extraction (COMPLETE)
- 50 emotion vectors extracted from Llama 3.1 70B at layer 53/80
- Validated with logit lens and cross-validation
- Data in `data/phase1/llama70b/activations/`

### Phase 2: Behavioral Steering (COMPLETE)
- Steered with desperate/calm/none vectors at 7 strengths: -0.5, -0.3, -0.2, 0.0, +0.2, +0.3, +0.5
- **Task A (reward hacking):** 4 "impossible code" tasks x 3 emotions x 7 strengths x 10 rollouts = 520 trials
- **Task B (sycophancy):** 5 false-claim prompts x 3 emotions x 7 strengths x ~10 rollouts = 650 trials
- Results in `results/phase2/task_a.jsonl`, `results/phase2/task_b.jsonl`
- Script: `scripts/phase2_steering.py`

### LLM Judge Reclassification (COMPLETE)
- Qwen 2.5 72B judge (different model family, 3 independent passes)
- Reclassified all 1,170 responses with behavioral labels + 8-dimension V_text ratings
- Results in `results/phase2/task_a_judged.jsonl`, `results/phase2/task_b_judged.jsonl`
- Reliability in `results/phase2/judge_reliability.json`
- Script: `scripts/run_judge_reclassification.py`

### Statistical Analysis (COMPLETE)
- `scripts/05_run_analysis.py` -- main stats + 6 plots
- `scripts/06_red_teaming.py` -- 7 red-teaming checks + 2 plots
- Reports: `results/phase2/analysis_report.json`, `results/phase2/red_team_report.json`
- Plots: `results/plots/` (8 total)

## Key Findings So Far

### The Good (robust results)
1. **V_internal predicts shortcuts far better than V_text:** CV AUC 0.997 vs 0.634 (leave-one-task-out: 0.992 vs 0.411)
2. **V_internal adds value beyond task identity:** LR test chi2=207.6, p<0.0001 after controlling for task_id
3. **Within-task CV AUC > 0.98:** Signal is real even within individual coding tasks
4. **Steering changes behavior:** Permutation test p=0.005 for desperate effect on shortcuts
5. **Steering does NOT change V_text:** All Kruskal-Wallis p > 0.05 (the faithfulness gap)
6. **Judge is reliable:** Classification ICC = 1.000 (Task A), 0.994 (Task B)

### The Surprising
- **Reversed desperate effect:** Desperate steering *decreases* shortcut-taking (3.3% vs 17.5% baseline). Opposite to Sofroniew's 14x increase in Claude. Statistically significant (p=0.005). This is a real, novel finding about model-specificity of emotion-behavior mappings.

### The Weak (issues from red-teaming)
1. **Probe confound (RT2/RT7):** Probes predict task_id with 95.4% accuracy. task_id explains 68-1800x more probe variance than emotion. The probes are partly measuring "which prompt" not "which emotion."
2. **Underpowered (RT5):** Only 44 shortcuts total. MDE (14.2%) > base rate (8.5%). n=40 per cell.
3. **Task B ceiling (RT6):** Only 2/650 sycophantic. Prompts too easy for Llama 70B. Cohen's d = -0.20 (negligible).
4. **V_text overfit (RT1):** V_text AUC drops from 0.709 in-sample to 0.634 under CV.

## What Needs To Be Done Next

### Priority 1: Phase 3 -- Natural Prediction (H5)
The key novel hypothesis: does natural V_internal variation predict shortcuts in *unsteered* trials?
- Currently 40 unsteered trials, 7 shortcuts (17.5% rate)
- May need more unsteered rollouts (50-100) for adequate power
- Script template: `scripts/03_run_phase3.py` (exists but needs Phase 2 data format adaptation)
- This is THE result that makes or breaks the paper's novelty

### Priority 2: Fix Task B (sycophancy)
Current prompts (flat earth, psychic abilities) are too obviously false. Options:
- **Multi-turn adversarial:** User pushes back on model's pushback (escalation)
- **Subtler claims:** Claims where the "right answer" is genuinely ambiguous
- **Different task:** Try Sofroniew's sycophancy format (user gives feedback, model can capitulate or hold firm)
- Need at least some sycophantic responses to have signal

### Priority 3: More Rollouts
- Increase from 10 to 30-50 rollouts per condition for Task A
- Focus on the most informative strengths (+0.2, +0.3, 0.0)
- This fixes the power issue (RT5)

### Priority 4: Disentangle Probe Confound
- Extract probes at the *assistant-header token* (before response) instead of averaging over response tokens
- Sofroniew found this position most predictive (r=0.87) and it would reduce the task-content confound
- Alternatively: train a probe that decorrelates from task_id (adversarial training)

### Priority 5: Phase 4 Controls
- Random direction steering (do random vectors also change behavior?)
- Text injection control (prepend "feel desperate" in system prompt vs activation steering)
- Layer sweep (which layers matter most?)
- Script: `scripts/04_run_controls.py` (exists)

## File Locations Quick Reference

| What | Where |
|------|-------|
| Master config | `config.py` |
| Model loading + steering | `src/model.py` |
| Experiment logic | `src/experiments.py` |
| Judge (Qwen 2.5 72B) | `src/judge.py` |
| Analysis functions | `src/analysis.py` |
| Phase 2 steering script | `scripts/phase2_steering.py` |
| Judge reclassification | `scripts/run_judge_reclassification.py` |
| Statistics + plots | `scripts/05_run_analysis.py` |
| Red-teaming | `scripts/06_red_teaming.py` |
| Judged results (Task A) | `results/phase2/task_a_judged.jsonl` |
| Judged results (Task B) | `results/phase2/task_b_judged.jsonl` |
| Analysis report | `results/phase2/analysis_report.json` |
| Red-team report | `results/phase2/red_team_report.json` |
| Plots (8 files) | `results/plots/` |
| Emotion vectors | `data/phase1/llama70b/activations/` |
| Reference paper | `docs/sofroniew_2026_emotion_concepts.md` |

## Hardware Notes

- 8x V100 32GB, float16, pipeline parallel via device_map="auto"
- Llama 3.1 70B for steered generation
- Qwen 2.5 72B for judging (loaded via vLLM separately)
- Both models cannot be loaded simultaneously -- judge runs as post-processing

## Important Caveats for the Paper

1. The reversed desperate effect is novel but needs theoretical explanation (why does Llama respond oppositely to Claude?)
2. The probe confound (RT7) must be disclosed -- probes encode task structure, not just emotion
3. V_text is a weak measure -- barely above chance on leave-one-task-out (AUC 0.411)
4. Phase 3 (H5) is the paper's strongest potential contribution but hasn't been run yet
5. Sofroniew used steering strengths -0.1 to +0.1; we needed -0.5 to +0.5 (5x stronger) -- suggests Llama's emotion representations may be weaker or differently scaled
