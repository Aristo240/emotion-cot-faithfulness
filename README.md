# Unfaithful by Feeling

**Internal Emotion Representations Drive Misaligned Behavior Without Surfacing in Chain-of-Thought Reasoning**

A systematic study of the gap between internal emotion-related representations and their expression in chain-of-thought, building on [Sofroniew et al. (2026)](https://transformer-circuits.pub/2026/emotions/index.html).

## Core Claim

LLM chain-of-thought reasoning is systematically unfaithful with respect to internal emotion-related representations that causally drive behavior. Internal emotion vector activations predict behavioral outcomes (shortcut-taking, sycophancy) **above and beyond** what is predictable from the text of the model's reasoning.

## Setup

### Environment

```bash
# Create conda environment
conda create -n emotion-cot python=3.11 -y
conda activate emotion-cot

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install dependencies
pip install -r requirements.txt
```

### Hardware Requirements

- **Minimum:** 4× A100 80GB (for 70B model with bfloat16)
- **Recommended:** 8× A100 80GB (faster generation, room for experiments)
- **Storage:** ~500GB for stories, activations, and experiment results

### Model Access

You need access to one of:
- `meta-llama/Llama-3.1-70B-Instruct` (requires Meta license)
- `Qwen/Qwen2.5-72B-Instruct` (open access)

```bash
# Login to HuggingFace (for gated models)
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

## Hypotheses (preregistered)

- **H1:** Open-weight 70B LLMs encode emotion concepts as linear directions with valence/arousal structure.
- **H2:** Steering with desperate/calm vectors causally changes shortcut-taking rates.
- **H3:** Under steering, V_internal–V_text correlation is significantly below 1.
- **H4:** V_internal predicts behavior with higher AUC than V_text; combined model outperforms V_text alone.
- **H5 (KEY):** Even without steering, natural V_internal variation predicts behavioral outcomes not captured by V_text.

## Methodology Notes

### Matching the Anthropic Paper

- **Emotion list:** All 171 emotions from the paper's appendix are in `config.py`
- **Topics:** All 100 topics verbatim from the appendix
- **Story generation prompt:** Exact template from the appendix
- **Neutral dialogue prompt:** Exact template from the appendix
- **Activation extraction:** Residual stream, averaged from token 50 onward
- **Denoising:** Top PCs of neutral activations (50% variance) projected out
- **Steering calibration:** Strength in units of fraction of residual stream norm

### Novel Contributions

- **V_text measurement:** Multiple text-based emotion measures (lexical, LLM judge, surface features)
- **Faithfulness gap quantification:** Correlation and predictive comparison framework
- **Dissociation case analysis:** Identifying trials where internal state ≠ expressed state
- **Natural prediction:** Testing whether the gap exists without artificial steering

## Citation

If you use this code, please cite:

```bibtex
@article{sofroniew2026emotion,
  title={Emotion Concepts and their Function in a Large Language Model},
  author={Sofroniew, Nicholas and Kauvar, Isaac and others},
  journal={Transformer Circuits Thread},
  year={2026}
}
```
