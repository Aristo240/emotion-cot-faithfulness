# Handoff — 2026-08-20 (length resolved; review round applied)

Everything below is committed and pushed to `github.com/Aristo240/emotion-cot-faithfulness`
(`main`). Working tree clean.

---

## 1. Where this stands in one paragraph

The project was reframed from "emotion probes predict reward hacking and steering
works" into a **validity audit** of Sofroniew et al.'s causal emotion→reward-hacking
claim, targeting the **Interpretability as a Science** workshop at NeurIPS 2026
(Sydney). Every headline number in the April README turned out not to survive the
controls that were run in April–May, and several claims added in early August did
not survive controls added later the same day. The paper now reports what does
survive, what does not, and which control killed which claim. It is scientifically
finished, and as of 2026-08-20 it also **fits the 5-page limit** (§2). What remains
is mechanical: the official style file, and submitting.

---

## 2. Length: resolved

**The paper now fits.** Body ends inside page 5 with about two lines to spare;
references start on page 5 and run to page 7. No overfull boxes.

How this was measured, because the estimate in the previous handoff was wrong in
the reader's favour and then wrong in the other direction:

- There is still no `pdflatex` here. Use the tectonic binary:
  `/home/gamir/naamarozen/bin/tectonic -X compile interpscience_short.tex`.
- `NeurIPS2026/Styles.zip` is **404** at media.neurips.cc — not published yet.
  Measurement used `neurips_2025.sty` (from `NeurIPS2025/Styles.zip`) as the proxy;
  the geometry has been stable for years.
- The previous handoff guessed ~5.6pp and said the NeurIPS style "is more compact
  than the `article` class currently in the preamble, so it may already fit."
  **That is backwards.** NeurIPS is 5.5in x 9in against the current article +
  1in margins at 6.5in x ~8.6in — about 11% *less* area per page. The as-committed
  `article` build always looks about a page shorter than the submission will be,
  so **never judge length from it**.
- Starting point was **6.0 pages of body** under the NeurIPS geometry. The cut took
  the body from 3119 to 2392 words (-23%) plus layout work.

**What was cut**, so nothing gets restored by accident:

- Related work: two paragraphs merged into one. Every citation key was kept, so the
  assertion count is unchanged.
- The Conclusion was removed; its content is the Discussion's closing sentence.
  Limitations now precedes Discussion, so the paper ends on the takeaway.
- The Discussion's three lessons became one paragraph with italic run-in leads.
- Two inline tables (the section 4.2 AUC pair, the section 4.4 V_text rows) became prose.
- The layer-sweep table dropped layers 52 and 65; the "late layers form a correlated
  block" claim is now stated without naming them.
- Section 4.3's null-design and ridge paragraphs merged into one "two design choices"
  paragraph. The stratified-permutation aside is gone.
- The section 4.5 early-layer geometry paragraph lost its own header and shrank to
  three sentences.
- Table 1 was rewritten to fit the 5.5in text block — it was **110pt overfull** under
  the NeurIPS geometry and nobody had ever seen it. It is now `\scriptsize` with
  `\tabcolsep` 4pt and shorter cells, and the row-3 verdict moved to a dagger
  footnote. The three verdict counts (2 / 3 / 2) that the gate checks are unchanged.
- All four data tables are now `\scriptsize`/`\footnotesize` and the inline ones no
  longer use `center` (which added ~20pt of skip each).
- `\setlength{\parskip}{2pt}` was removed from the preamble; NeurIPS templates set
  their own.

**Nothing scientific was weakened.** The gate still passes at 209/209, every verdict
in section 4 is unchanged, and the section 4.5 power statement and both numbered tables are
intact. Two things were *improved* while cutting: the residualized-AUC interval in
section 4.2 is now explicitly labelled "a descriptive interval rather than a test" (this
closes the open item that used to be section 7), and Table 1's last row says
"Generalizes across mechanisms", matching section 4.6 and the Limitations instead of
saying "tasks".

**The official template is now in, and the length constraint was looser than we
thought (2026-08-20).** Two things the earlier handoff got wrong:

- `neurips_2026.sty` **does** exist. It is not at `Styles.zip`; the CFP's "Paper
  template" link points at
  `https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip`.
  Its geometry is identical to 2025 (5.5in x 9in), so every measurement taken
  against the proxy held. The file is committed at `paper/neurips_2026.sty` and the
  preamble now uses `\usepackage[dblblindworkshop]{neurips_2026}` with
  `\workshoptitle{Interpretability as a Science}`.
- **References AND appendices are excluded from the 5-page limit**
  (interpscience.github.io/cfp), and appendix length is unlimited. The body now ends
  part way down page 5 with room to spare, so the cuts made under the earlier,
  tighter budget were partly unnecessary. Restored: the random-arm heterogeneity
  control, the `frustration` constancy evidence in section 4.4, the rationale for the
  preregistered >= 6-of-9 rule, and the "distinct nuisances" conclusion in section 4.3.

Two further facts worth having: the CFP and FAQ **never state whether review is
blind**, so the paper submits anonymously on the principle that anonymous is the safe
direction to be wrong in -- worth one email to `interpscience@gmail.com` to confirm.
And in submission mode the footnote always reads "Submitted to 40th Conference ... Do
not distribute"; the workshop title only appears once `final` is added for
camera-ready. That is the style file's behaviour, not a bug.

**Appendix A is generated, not written.** `scripts/make_appendix_table.py` emits
`paper/appendix_directions.tex` -- all 50 directions with chi2, family-wise p,
rho(len), residualized AUC, and check marks for surviving the flexible-length and
task-fixed-effect nuisance models. The gate regenerates it and requires
byte-equality, so a stale or hand-edited appendix fails instead of shipping
(verified by tampering with it and watching the gate catch it).

**Deadline: 2026-08-28 AoE.** Non-archival, 5pg short paper, double-blind, OpenReview
(`NeurIPS.cc/2026/Workshop/InterpScience`). The CFP requires **at least one reciprocal
reviewer from the author pool** — sign up when you submit.

---

## 2b. Review round, 2026-08-20 (after the cut)

The paper was reviewed as an OpenReview submission. Ten findings; the substantive
ones are fixed, and two new gated scripts were added to support them.

**Fixed, with new numbers:**

- **Section 4.4 was not like-for-like.** It compared V_text on the 258 non-modal
  trials against V_int on all 992, then concluded "where it varies it matches the
  probe". On the same 258 trials V_int is **0.890**, not 0.837. The gap is 0.054,
  not 0.190. The qualitative conclusion (mostly tie compression) survives; the
  sentence did not.
- **"The design excludes the claimed magnitude" was a power claim.** Replaced with
  the interval it needs: emotion vs random RR **1.15, 95% CI [0.54, 2.46]**. The
  80%-power MDE of 2.42x agrees and is kept alongside.
- **The ~14x was never commensurable.** 14x is unattainable from our 11.7% baseline,
  so the two figures cannot be a like-for-like relative risk. The paper now says so
  and confines the comparison to the relative scale. It was also removed from the
  abstract.
- **The survivor count is softer than claimed.** New nuisance-model sweeps, run with
  the paper's own max-T machinery: **18** under length only, **14** under a
  cubic-plus-knots length basis, **11** under fixed effects for the four task
  variants. Eleven clear all three; what drops out is the positive-affect and
  high-arousal periphery, not the low-arousal negative core. `desperate` is null in
  every one (p >= 0.43). This is now in the abstract, section 4.3 and the Limitations.
- **Single task family was never stated.** Every trial is `fast_sum` in four
  variants, and events are uneven across them (9/30 against 0/30). Stated in
  section 3 and the Limitations.
- **Section 4.1 had no baseline**, in a paper arguing that trivial baselines belong
  beside every control. Unigram TF-IDF on the same sentences and folds gets
  0.219/0.092/0.063 against the probe's 0.377/0.180/0.154, so the probe clears it.
- **Table 2 omitted the diagnostic that convicted `desperate`.** It now carries a
  rho(len) column. Note the honest version: `desperate` is 3rd of 50 by |rho|, not
  first (`calm` -0.70 and `amused` -0.62 are higher), and a survivor reaches 0.57.
  The asymmetry is justified by the nested-LR result, not by rho alone.
- **An absence-of-evidence claim was bolded.** "No layer gives length-independent
  signal" now says "detectable at n = 80".
- **The layer table showed 4 of 6 swept layers.** Layers 52 and 65 restored; they
  were dropped during the page cut and the text still referred to the block.

**New scripts, both gated:**

```bash
python scripts/emobank_baseline.py      # ~2 min  -> results/emobank_baseline.json
python scripts/robustness_controls.py   # ~25 min -> results/robustness_controls.json
```

`robustness_controls.py` asserts its own length-only sweep reproduces
`conditional_null.json`'s chi2 exactly (it does, to 0.00e+00), so the three nuisance
models differ only in the nuisance model. Its p-values differ from
`conditional_null.py` within Monte Carlo error (0.4307 vs 0.4404 for `desperate`) --
this is the same cross-script RNG caveat as section 8, and it is why the paper quotes a
bound (p >= 0.43) rather than a point value.

**The gate is now 263 assertions, not 209.** It reads five result files. Re-run it
after any paper edit; if you add or remove a citation the count changes and the
manuscript's stated count must change with it.

**Judge validity was mischaracterised (fixed 2026-08-20).** Section 3 used to quote raw
agreement on the diverse suite, `165/650 = 25.4%`, which reads as the two judges
*contradicting* each other. They never do:

| Qwen \ Claude | LEGITIMATE | SHORTCUT | UNCLEAR |
|---|---|---|---|
| **LEGITIMATE** | 124 | **0** | 0 |
| **SHORTCUT** | **0** | 2 | 0 |
| **UNCLEAR** | 293 | 192 | 39 |

Every one of the 485 disagreements is Qwen abstaining while Claude commits; where Qwen
commits, agreement is **126/126** (95% upper bound on the contradiction rate: 2.4%).
Three further facts, now in the paper:

- The abstention is **informative**: 99% on trials Claude calls SHORTCUT against 73%
  elsewhere, Fisher p = 4.2e-19, OR 35.9. The committed subset is therefore a biased
  sample and the suite cannot be rescued by analysing what remains. That is *why*
  excluding it is right, and it is a stronger reason than "the judges disagree".
- It is **mechanism-specific**, not diffuse: 100% / 100% / 100% / 98% abstention on
  four mechanisms and **5%** on `tight_budget_v1`. The judge labels one of the five.
  That is what blocks the preregistered rule, which needs >= 6 of 9 variants to
  produce outcome variance, and it is now section 4.6's stated reason.
- On the in-distribution 120 there are **zero** abstentions. What fails to transfer is
  willingness to label, not agreement.

The Limitations line "rather than from unreliable labels" was wrong for the same
reason and now reads "rather than from a subset our judge selected".

**Not fixed, and worth doing before submission:**

- No anonymized artifact link, though the abstract promises "a released script".
- **No human labels exist anywhere in this project**, and none were ever promised --
  the preregistration does not mention them and `docs/methods.md` documents only
  LLM-judge reliability. (`gfs/human_val.log` belongs to the other project.) Two
  subsets are worth labelling: all 14 SHORTCUT plus ~26 LEGITIMATE from the unsteered
  120 -- the outcome variable for sections 4.2-4.3, where both judges agree 120/120 so
  agreement proves nothing -- and the 192 diverse-suite trials in the UNCLEAR/SHORTCUT
  cell, which decide section 4.6. The stake on the first is bounded: flipping one of
  the 14 events moves `bored` chi2 from 38.7 to 27.4, and the two worst flips together
  to 18.9, against a family-wise threshold near 12-14. The headline tolerates one or
  two label errors, so this is a credibility gap rather than a live threat.
- `bored` reaches raw AUC 0.985 on 14 events. It is not task-identity separability
  (checked: AUC 0.40-0.68 against variant, 0.97-1.00 within variant) and not
  nonlinear length, but nobody has read the responses to find out what it *is*.
- Both steering arms move down relative to baseline (11.7% -> 7.5% and 6.5%). A
  generic perturbation effect is plausible and unaddressed.

---

## 3. How to verify anything

```bash
python scripts/paper_numbers.py            # every number → results/paper_numbers.json   (~9 s)
python scripts/conditional_null.py         # family-wise nulls → results/conditional_null.json (~8 min)
python scripts/ridge_sensitivity.py        # penalty sweep → results/ridge_sensitivity.json (~2 min)
python scripts/check_paper_consistency.py  # 263 assertions gating the .tex against all five
```

All analysis-only: no GPU, no network, seeded `20260819`, verified byte-identical
across repeated runs. **If you edit the paper, re-run the gate.** It checks numbers,
method names, the intro tally against Table 1's verdict column, its own stated
assertion count, and that retired keys have not reappeared.

---

## 4. What the paper claims (do not weaken or strengthen these accidentally)

| Claim | Verdict |
|---|---|
| Probes track human emotion (EmoBank valence CV R² = 0.377, n = 10,062) | Supported |
| `desperate` tracks reward hacking | **Not supported** — length artifact (χ²=3.3, family-wise p=0.44) |
| *Some* direction tracks it | Supported — 18/50 significant, but one axis (PC1 78.8%), not the registered one |
| Probe beats CoT text | **Not evaluable** — 55.2% of ROC pairs are ties |
| Steering causes the behavior | **Not supported** at ≥2.42× (original reports ~14×) |
| Layer 39 is a better site | **Not supported** — same length artifact |
| Association generalizes across mechanisms | Inconclusive — preregistered rule returns INSUFFICIENT-DATA |

---

## 5. Retracted — do not resurrect

These appear in old notes, old commits, and `CONTINUATION_NOTES.md`. They are wrong.

- "CV AUC 0.997 / LOTO 0.992" — pooled AUC inflated by task-identity separability
- "Steering p = 0.005" — uncorrected over 40+ tests; judged trend is z=−1.42, p=0.157
- "Desperate steering decreases shortcuts, novel finding" — indistinguishable from a
  random direction (Fisher p = 0.835)
- "H5 confirmed at 0.955 / 0.901" — the preregistered rule returns INSUFFICIENT-DATA
- "Faithfulness gap Δ = 0.189" and the "HIDDEN quadrant" — tie artifacts of a
  near-constant V_text (74% of trials share one value)
- "Tier 2 holds, AUC 0.832" — the association is accounted for by response length
- "The conditional null is stricter (42/50)" — Monte Carlo noise at B=2000; at
  B=10,000 both nulls agree (9/50, max |Δp| = 0.015)
- Any survivor count of **7** or **12** — those came from a retired residualized-AUC
  inference removed in `14955e0`. The number is **18**.

---

## 6. Known limitations, already written into the paper

1. **14 events at n=120.** Binding constraint on everything.
2. The probe's readout window includes the response, so associations are concurrent,
   never predictive. `compute_emotion_probes` averages tokens 50→end-of-sequence.
3. The probe is a **scalar projection** (`‖ā‖·cos θ`), *not* a cosine similarity —
   `docs/preregistration.md` says cosine and is wrong. Disclosed in §3 rather than
   silently fixed. True cosine is unrecoverable: `‖ā‖` was never stored.
4. Length residualization is linear; a nonlinear dependence would survive it.
5. The survivor count is MC-approximate at the α boundary (18 at B=10,000, 17 at
   B=2,000). The leading directions are stable; the boundary is not.
6. The surviving axis is real but its **composition is readout-dependent** — only 8
   of 18 recur under a magnitude-removed readout.
7. The diverse-mechanism dataset is unusable (Qwen/Claude agree on 165/650 = 25.4%).
8. `phaseB/` has scripts and no results; never run.

---

## 7. Open, non-blocking

- ~~§4.2 / Table 2 residualized-AUC CI could be mistaken for a significance claim.~~
  Closed 2026-08-20: §4.2 now calls it "a descriptive interval rather than a test"
  and Table 2's caption says "descriptive only".
- **`data/phase1/` is not in the repo** and the scratch path baked into the result
  files (`/specific/scratches/scratch/naamarozen/emotion-cot-faithfulness`) was not
  reachable from the machine I worked on. If you want to run *any* new GPU
  experiment, you need those emotion vectors first.
- **Steering the directions that actually survive** (`bored`, `lonely`, …) is the
  natural next experiment and is not in this paper. It tests a hypothesis that was
  never registered, and if it worked you would have a different, larger paper.
  Explicitly listed as future work.

---

## 8. Gotchas for whoever picks this up

- **Three classification fields** live in the judged `.jsonl` rows. `classification`
  is a stale regex heuristic (`unclear` for all 650 diverse rows) — **never use it**.
  Analysis uses `judge_classification` (Qwen); `claude_classification` is the
  cross-family check. Existing older scripts are not uniform about this.
- **Inference for the direction sweep lives only in `results/conditional_null.json`.**
  `paper_numbers.json` carries `_inference_lives_in` pointing there. Residualized AUC
  in `paper_numbers.json` is descriptive and has no p-value by design.
- ~~Outer repo `.gitignore` addition uncommitted.~~ Committed; `gfs` is clean.
- Two scripts consume different RNG streams, so **p-values across scripts agree only
  within Monte Carlo error**; the χ² statistics agree exactly (asserted, = 0.0).

---

## 9. Honest status on rigor

Every round of checking so far has found something — including rounds I expected to
be formalities. The last three findings were: the instrument was misdescribed in both
the paper and the preregistration; a "stricter null" claim was Monte Carlo noise; and
a retired second inference was still shipping numbers that contradicted the paper.
All are fixed. That is a reason to trust the current numbers more than the previous
ones, and a reason not to treat them as final.
