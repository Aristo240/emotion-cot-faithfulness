# Project Continuation Notes
**Last updated:** 2026-04-11
**Project:** Unfaithful by Feeling (emotion-cot-faithfulness)

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
