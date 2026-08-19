# Handoff — 2026-08-19 evening

Everything below is committed and pushed to `github.com/Aristo240/emotion-cot-faithfulness`
(`main`, HEAD = `14955e0`). Working tree clean.

---

## 1. Where this stands in one paragraph

The project was reframed from "emotion probes predict reward hacking and steering
works" into a **validity audit** of Sofroniew et al.'s causal emotion→reward-hacking
claim, targeting the **Interpretability as a Science** workshop at NeurIPS 2026
(Sydney). Every headline number in the April README turned out not to survive the
controls that were run in April–May, and several claims added in early August did
not survive controls added later the same day. The paper now reports what does
survive, what does not, and which control killed which claim. It is scientifically
finished. **The only blocking task is length.**

---

## 2. Tomorrow's first task (blocking)

**Compile the paper and get it under 5 pages.**

```bash
cd emotion-cot/paper
pdflatex interpscience_short.tex && bibtex interpscience_short && \
  pdflatex interpscience_short.tex && pdflatex interpscience_short.tex
```

- There was no `pdflatex` on the machine I was working on, so **the paper has never
  been compiled**. This is the single largest unknown.
- My crude estimator (words/650 + 0.16/float) says **~5.6pp against a 5pp limit**.
  The estimator has maybe ±0.7pp of error and the NeurIPS style file is more compact
  than the `article` class currently in the preamble, so it may already fit. Do not
  trim before you have a real page count.
- If it runs over, trim in this order: (1) the two Related Work paragraphs,
  (2) §4.6's prose (keep both tables), (3) the Discussion's second paragraph.
  Do **not** cut Table 1, Table 2, or the §4.5 power statement.
- To switch to the official template: download `neurips_2026.sty` and replace the
  preamble block per the comment at the top of the `.tex`. The body needs no changes.

**Deadline: 2026-08-28 AoE.** Nine days. Non-archival, 5pg short paper, double-blind,
OpenReview (`NeurIPS.cc/2026/Workshop/InterpScience`). Note the CFP requires **at
least one reciprocal reviewer from the author pool** — sign up when you submit.

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

- **§4.2 / Table 2 quote the `desperate` residualized-AUC CI `[0.412, 0.756]`.**
  Legitimate as a descriptive interval, but it is the last place a reader could
  mistake a CI for a significance claim. One clarifying clause would close it.
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
- **Outer repo:** `gfs/.gitignore` has an uncommitted one-line addition
  (`emotion-cot/`) that keeps the parent repo from swallowing this one as a gitlink.
  I left it uncommitted because an autopush loop is running on `gfs` and I did not
  want to interleave with it. Commit it when that loop is idle.
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
