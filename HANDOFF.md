# Handoff — 2026-08-20 (length resolved)

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

**Before submitting:** drop in the real `neurips_2026.sty` when it appears, swap the
preamble per the comment at the top of the `.tex`, and re-measure. There are only
about two lines of slack, so if the 2026 style differs at all, re-check before adding
anything back.

**Deadline: 2026-08-28 AoE.** Non-archival, 5pg short paper, double-blind, OpenReview
(`NeurIPS.cc/2026/Workshop/InterpScience`). The CFP requires **at least one reciprocal
reviewer from the author pool** — sign up when you submit.

---

## 3. How to verify anything

```bash
python scripts/paper_numbers.py            # every number → results/paper_numbers.json   (~9 s)
python scripts/conditional_null.py         # family-wise nulls → results/conditional_null.json (~8 min)
python scripts/ridge_sensitivity.py        # penalty sweep → results/ridge_sensitivity.json (~2 min)
python scripts/check_paper_consistency.py  # 209 assertions gating the .tex against all three
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
