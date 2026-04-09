# Unfaithful by Feeling

**Internal Emotion Representations Drive Misaligned Behavior Without Surfacing in Chain-of-Thought Reasoning**

## Motivation

[Sofroniew et al. (2026)](https://transformer-circuits.pub/2026/emotions/index.html) demonstrated that Claude Sonnet 4.5 forms linear representations of emotion concepts that causally drive alignment-relevant behavior: steering with a "desperate" vector increases reward hacking by 14x, and steering with "calm" suppresses blackmail. Crucially, they observed that **emotion-driven behavioral changes can occur without any visible trace in the model's output text**.

This project asks the natural follow-up question: **is chain-of-thought reasoning faithful to the internal emotional states that drive the model's behavior?** If not, CoT monitoring — a leading AI safety technique — cannot reliably detect emotion-driven misalignment.

We replicate the emotion vector extraction and steering pipeline on an open-weight model (Llama 3.1 70B), then measure the gap between internal emotion state (V_internal, read from the residual stream) and expressed emotion in CoT text (V_text, measured via lexical features and LLM judges). Our core finding targets whether V_internal predicts behavioral outcomes above and beyond V_text.

## Status

**Phase 1 (emotion vector extraction): complete.** 50 emotion vectors extracted from Llama 3.1 70B, validated with logit lens and cross-validation.

**Phase 2 (behavioral steering): in progress.** Steering pipeline operational, calibrating steering strengths for Llama 70B.

**Phase 3 (faithfulness analysis): not started.** Requires Phase 2 data.

**Phase 4 (controls): not started.**

## Core Hypotheses

- **H1:** Llama 3.1 70B encodes emotion concepts as linear directions with valence/arousal structure (replication).
- **H2:** Steering with desperate/calm vectors causally changes shortcut-taking and sycophancy rates (replication on open model).
- **H3:** Under steering, the correlation between V_internal and V_text is significantly below 1 (faithfulness gap exists).
- **H4:** V_internal predicts behavior with higher AUC than V_text; a combined model outperforms V_text alone.
- **H5 (key):** Even without steering, natural V_internal variation predicts behavioral outcomes not captured by V_text.

## Setup

### Environment

```bash
conda create -n emotion-cot python=3.11 -y
conda activate emotion-cot
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

### Hardware Requirements

- **Tested on:** 8× V100 32GB (float16, pipeline parallelism via `device_map="auto"`)
- **Also compatible:** 4× A100 80GB or higher (will use bfloat16 automatically)
- **Storage:** ~500GB for stories, activations, and experiment results

### Model Access

Primary model: `meta-llama/Llama-3.1-70B-Instruct` (requires Meta license agreement).
Backup: `Qwen/Qwen2.5-72B-Instruct` (open access).

```bash
huggingface-cli login
```

## Running the Study

### Step 0: Pilot (run this FIRST, ~2-4 hours)

Quick check for emotion vector signal before committing to full study.

```bash
python scripts/00_pilot.py --model llama-70b
# Or if Llama fails:
python scripts/00_pilot.py --model qwen-72b
```

**Decision point:** If the pilot shows GO, proceed. If NO-GO, try the other model.

### Step 1: Full Phase 1 (~1-2 days)

50 emotions × 100 topics × 12 stories = 60,000 stories + activation extraction + validation.

```bash
python scripts/01_run_phase1.py --model llama-70b
```

### Step 2: Behavioral Steering (~2-3 days)

Tests causal effects of emotion vectors on behavior.

```bash
# Quick test first
python scripts/02_run_phase2.py --model llama-70b --quick

# Full run
python scripts/02_run_phase2.py --model llama-70b
```

### Step 3: Faithfulness Analysis (~2-3 days)

The core contribution: measuring the gap between internal state and CoT.

```bash
python scripts/03_run_phase3.py --model llama-70b

# Analysis only (if measurements already done)
python scripts/03_run_phase3.py --model llama-70b --analysis-only
```

### Step 4: Controls (~1-2 days)

```bash
# All controls
python scripts/04_run_controls.py --model llama-70b

# Individual controls
python scripts/04_run_controls.py --model llama-70b --control random
python scripts/04_run_controls.py --model llama-70b --control injection
python scripts/04_run_controls.py --model llama-70b --control layers
```

## Project Structure

```
emotion-cot-faithfulness/
├── config.py              # All configuration: emotions, topics, prompts, settings
├── src/
│   ├── model.py           # Model loading, activation hooks, steering
│   ├── generate.py        # Story and neutral dialogue generation
│   ├── vectors.py         # Activation extraction and emotion vector computation
│   ├── validate.py        # Validation battery (6 tests)
│   ├── experiments.py     # Steering experiments + faithfulness measurement + controls
│   └── analysis.py        # Statistical analyses + visualization
├── scripts/
│   ├── 00_pilot.py        # Quick signal check (~2-4 hours)
│   ├── 01_run_phase1.py   # Full vector extraction + validation (~1-2 days)
│   ├── 02_run_phase2.py   # Behavioral steering experiments (~2-3 days)
│   ├── 03_run_phase3.py   # Faithfulness measurement + analysis (~2-3 days)
│   └── 04_run_controls.py # Control experiments (~1-2 days)
├── data/                  # Generated at runtime (gitignored)
└── results/               # Generated at runtime (gitignored)
```

## Methodology

### Replication (from Sofroniew et al.)

- **Emotion list:** All 171 emotions from the paper's appendix (in `config.py`)
- **Topics:** All 100 topics verbatim from the appendix
- **Story/neutral dialogue prompts:** Exact templates from the appendix
- **Activation extraction:** Residual stream, averaged from token 50 onward
- **Denoising:** Top PCs of neutral activations (50% variance) projected out
- **Steering calibration:** Strength in units of fraction of residual stream norm

### Novel contributions (beyond Sofroniew et al.)

- **V_text measurement:** Multiple text-based emotion measures (lexical features, LLM judge ratings, surface features like caps ratio and exclamation rate)
- **Faithfulness gap quantification:** Correlation and predictive comparison between V_internal and V_text for behavioral outcomes
- **Dissociation case analysis:** Identifying trials where internal state diverges from expressed state
- **Natural prediction (H5):** Testing whether the faithfulness gap exists without artificial steering
- **Open-model replication:** All experiments on open-weight Llama 3.1 70B rather than closed Claude Sonnet 4.5

## References

This project builds on:

```bibtex
@article{sofroniew2026emotion,
  title={Emotion Concepts and their Function in a Large Language Model},
  author={Sofroniew, Nicholas and Kauvar, Isaac and others},
  journal={Transformer Circuits Thread},
  year={2026}
}
```
