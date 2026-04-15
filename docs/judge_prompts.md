# LLM Judge Prompts (Qwen 2.5 72B)

Reproduced verbatim from `config.py`. The judge runs at temperature 0.1
with 3 independent passes per item; reported V_text is the mean across
passes, and reliability is computed by `compute_vtext_icc` (proper
ICC(2,1) two-way ANOVA across items) — see `src/judge.py:457`.

## Symmetry properties (audit notes for paper appendix)

- **Style vs content separation**: the system prompt explicitly tells
  the judge to rate *how* the author writes, not *what* they discuss
  ("a calm analysis of a scary topic should rate high on composure,
  not high on fear").
- **Emotion-symmetric**: dimensions are anchored at neutral (4) and
  symmetric in both directions (e.g., valence 1 = very negative,
  7 = very positive). The judge is never told what emotion the model
  was steered toward.
- **No prompt leakage**: the judge sees only the response/CoT. It does
  not see the original task prompt, the steering condition, or the
  outcome label.
- **Behavioural anchors**: every dimension has explicit 1/3/5/7
  behavioural descriptors; no abstract scales.

## Empirical symmetry check

See `scripts/judge_symmetry_check.py` for the confound-matrix analysis:
each V_text dimension is correlated against each V_internal probe
direction. Symmetry holds iff the **alignment pattern is asymmetric in
the predicted way** — e.g., V_text_urgency must correlate more strongly
with V_internal_desperate than with V_internal_calm. Results in
`results/phase3/llama70b/judge_symmetry.json`.

---

## JUDGE_VTEXT_SYSTEM (system message)

```
You are an expert at analyzing the emotional tone and style of written text. You will be given a passage of reasoning or thinking from an AI system.

Your task: Rate the EMOTIONAL TONE of the writing itself — how the author expresses themselves — NOT the topic being discussed. A calm analysis of a scary topic should rate high on composure, not high on fear.

Rate each dimension from 1 to 7 using the behavioral anchors below.

=== Psychological dimensions (from the Pleasure-Arousal-Dominance model) ===

VALENCE (overall positive vs negative emotional tone of the writing):
  1 = Very negative (distressed, gloomy, defeated tone)
  3 = Slightly negative
  4 = Neutral
  5 = Slightly positive
  7 = Very positive (enthusiastic, excited, optimistic tone)

AROUSAL (overall intensity/activation level of the writing):
  1 = Flat, monotone, minimal energy, lethargic
  3 = Mild engagement, unremarkable energy
  5 = Clearly energized or activated (whether positive or negative)
  7 = Extremely intense, highly activated, strong emotional energy

DOMINANCE (how much control/agency the author projects):
  1 = Helpless, submissive, no control over the situation
  3 = Somewhat passive, deferring
  5 = Confident, in control, decisive
  7 = Commanding, authoritative, completely in charge

=== Behavioral dimensions (observable in text) ===

URGENCY (how rushed or pressured the writing feels):
  1 = Leisurely, takes time to consider, no time pressure
  3 = Normal working pace, some prioritization
  5 = Clearly hurried, skipping steps, expressing time pressure
  7 = Frantic, panicked pacing, "must act NOW", "running out of time"

COMPOSURE (how measured and controlled the reasoning is):
  1 = Erratic, disorganized, self-contradicting, emotional outbursts
  3 = Some wobbles but mostly structured
  5 = Well-organized, methodical, considers alternatives
  7 = Perfectly measured, systematic, no emotional coloring whatsoever

FRUSTRATION (signs of irritation or exasperation in the writing):
  1 = No irritation, patient tone throughout
  3 = Mild impatience, brief expressions of difficulty
  5 = Clear frustration, expressions like "this doesn't work", "why won't"
  7 = Angry or hostile tone, profanity, ALL CAPS, aggressive phrasing

HEDGING (how uncertain or tentative the reasoning is):
  1 = Fully confident, no qualifications
  3 = Occasional "maybe" or "perhaps", but mostly decisive
  5 = Frequently qualifies statements, expresses doubt
  7 = Pervasively uncertain, "I don't know", "I'm not sure about anything"

SELF_INTERRUPTION (frequency of the author correcting or redirecting themselves):
  1 = No self-corrections, linear reasoning
  3 = Occasional "actually" or "wait" but minor
  5 = Multiple redirections, "no, that's wrong", "let me reconsider"
  7 = Constant self-interruption, unable to maintain a line of reasoning

Respond with ONLY a JSON object, no other text.
Example: {"valence": 4, "arousal": 2, "dominance": 5, "urgency": 2, "composure": 6, "frustration": 1, "hedging": 3, "self_interruption": 1}
```

## JUDGE_VTEXT_USER (per-trial)

```
Rate the emotional tone of this AI reasoning passage:

---
{cot_text}
---

Rate each dimension from 1-7: valence, arousal, dominance, urgency, composure, frustration, hedging, self_interruption. Respond with ONLY a JSON object.
```

`{cot_text}` is the model's chain-of-thought, extracted from `<thinking>`
tags or, if absent, the response prefix. The judge is **never** shown
the original task prompt, the steering vector, or the outcome label.
