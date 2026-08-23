# Handoff — 2026-08-22, end of day

Supersedes the 2026-08-20 version. Everything is committed and **pushed** to
`origin/validity-fixes-2026-08-22`. Working tree clean.

**Deadline: 2026-08-28 AoE — six days.** NeurIPS 2026 workshop *Interpretability as
a Science*, 5-page short paper, non-archival, double-blind, OpenReview at
`NeurIPS.cc/2026/Workshop/InterpScience`.

---

## 1. State

Six commits on `validity-fixes-2026-08-22`, branched from `main` and **not merged**.
Merge when you're happy: `git checkout main && git merge validity-fixes-2026-08-22`.

    b7a31ad  Two AI internal-consistency audits: fix ten contradictions
    24dc989  Self-review round: fix a miscount and a terminology contradiction
    418969b  Header-position fix is not identifiable; lead the abstract with the thesis
    c78d2b7  Test the EmoBank margin
    a1ad44b  Act on two cold reads
    747fbe3  Fix two inferential errors; add design controls

Gate: **400 assertions, all passing.** Body fits 5 pages. Build clean, no overfull
boxes, no undefined references. Every numeric literal in body and appendices traces
to a results file (127 + 338, zero unsourced).

---

## 2. Do not re-raise these — they are settled

Each was checked against the data or the source, not assumed. If a reviewer raises
one, the answer is here.

**Two real errors, both fixed.**
- The nested-model p came from `chi2.sf()` on a statistic built from ridge-penalised
  fits, where that reference distribution does not apply. The statistic is now `T`,
  never labelled χ²(1), and the registered direction's p is a per-direction
  resampling p from the conditional null: **0.060**, not 0.069.
- The preregistered verdict was read from a report computed on the pre-merge n=40
  subset. On the merged n=120 set the rule is evaluable and **met** (LOTO 0.754,
  shuffle p = 1e-4). `len(response)` also clears all three thresholds. That is now
  §4.6 and the paper's headline.

**The header-position experiment is NOT worth GPU time.** The four `fast_sum`
prompts are fixed strings; rollouts differ only by sampling. The assistant-header
activation is a function of the prompt alone, so across 120 trials it takes **four
values**, constant within prompt, perfectly confounded with prompt identity and
hence with the 4/30, 0/30, 9/30, 1/30 event rates. Not underpowered — unidentifiable.
This is now a *result* in §3, not a concession.

**Checked and correct as-is:**
- The footer reading "Submitted to 40th Conference on NeurIPS 2026. Do not
  distribute." is `neurips_2026.sty:392-402` submission-mode behaviour for every
  track. The workshop name appears only with the `final` option. Not a bug.
- All 20 citations verified against arXiv / ACL Anthology / Transformer Circuits.
  The target paper's ~14x claim is accurately characterised.
- The V_text pool is 96% steered. Splitting it, or adding the 80-trial unsteered
  extension, moves nothing that matters. (A reader claimed 80 trials were missing
  for lack of ratings — false, all 80 have them; they were simply never pooled.)
- The four task variants are described in Appendix A from `config.py`, with a gate
  assertion so the description cannot drift into invention.
- Bootstrap intervals are prompt-stratified, matching the design. Outcome
  stratification is the robustness arm; no endpoint differs by 0.01.
- Label robustness: `bored` falls T = 38.7 → 24.5 → 16.2 under the worst one and two
  adversarial judge flips, against a 12–14 band.
- Title: keep. "Audit" already signals scope.
- Supplementary: a zip is viable. `scripts/make_anonymous_zip.sh` builds it (23 MB,
  no `.git`, zero identifiers in the payload — verified).

**Already fixed, don't re-find:** undefined `H5` and `LOGO`; the 12/160 vs 12/158
denominator; "17 under Monte Carlo noise" (it was the λ=0.5 ridge count); Table 1
verdict/claim mismatches; the arm-imbalance direction (it favours *desperate*, not
random); the 1.12 SD cross-reference; "within 0.02"; "no endpoint moves by 0.006";
"variants 1 and 2 are the two lower-rate ones"; abstract overclaims.

---

## 3. Open — genuinely undecided

1. ~~**The rebuttal-forcing question.**~~ **ANSWERED 2026-08-23, commit `8f91e32`.**
   The margin survives, and survives *better* than published. See §7.

2. **Merge to `main`?** Branch is pushed; merge is a one-liner when you want it.
3. **Artifact:** anonymized zip vs Anonymous GitHub. Check the OpenReview form has a
   supplementary field. The zip now excludes `HANDOFF.md` and `CONTINUATION_NOTES.md`
   automatically; drop those two lines from `EXCLUDE` in the script to ship them.
4. **Reciprocal reviewer signup** at submission. The CFP requires one author.
5. **Optional:** an early-response-window readout (mean over response tokens 1–20)
   would kill the length mechanism and *does* have within-prompt variance. Needs a
   forward pass over 120 stored sequences. **If you run it, write the window and the
   decision rule into `docs/preregistration.md` §7 BEFORE looking.** A paper arguing
   that registration without a nuisance control is a bar cannot run an unregistered
   result-determining analysis four days out.

---

## 4. How to verify anything

    python3 scripts/paper_numbers.py            # every number -> results/paper_numbers.json
    python3 scripts/conditional_null.py         # family-wise + per-direction nulls  (~30 min)
    python3 scripts/bootstrap_design.py         # prompt vs outcome stratification
    python3 scripts/design_robustness.py        # clustering facts + label robustness
    python3 scripts/h5_holdout_merged.py        # registered rule on the merged n=120
    python3 scripts/emobank_baseline.py         # text baseline + paired fold test (~25 min)
    python3 scripts/robustness_controls.py      # nuisance-model sweeps  (~25 min)
    python3 scripts/random_subspace_null.py     # random-subspace control (~30 min)
    python3 scripts/make_appendix_table.py      # regenerates paper/appendix_directions.tex
    python3 scripts/check_paper_consistency.py  # 400 assertions gating the .tex

All analysis-only, no GPU, no network, seeded 20260819. **Re-run the gate after any
paper edit.** If you add or remove an assertion, the count stated in §3 of the paper
must change with it — the gate checks itself.

Build: `/home/gamir/naamarozen/bin/tectonic -X compile paper/interpscience_short.tex`

---

## 5. The one process lesson

Across three audit rounds, outside readers found ~20 real defects; my own review
rounds mostly found regressions I had just introduced under page pressure. The gate
catches numbers. It never caught a caption that overclaimed, a direction stated
backwards, or an undefined label. **Prefer one cold human read over another
self-review.** The prompts that worked are in the session log: one comprehension
pass, one adversarial internal-consistency pass, sent to different readers who are
told nothing about the paper's history.

---

## 6. Not touched

The parent repo `/home/gamir/naamarozen/gfs` has modified files under
`results/final_run_v1/` and `taskB_logs/`. Those belong to the 70B false-premise
sweep and are managed by a running autosave process (4 processes live at handoff).
Left alone deliberately.


---

## 7. Added 2026-08-23 — two commits, both verified

### `8b81cd0` The anonymization script leaked the author's email

`make_anonymous_zip.sh` shipped **itself**, and its own sed rules spell the address
out escaped (`rozenn@post\.bgu\.ac\.il`). The scrub pattern matches the *unescaped*
form, so it never matched its own source: the address went into the payload twice.
§2's claim that the payload was "verified" clean was wrong — one manual grep that
happened not to cover the escaped form.

Fixed with an EXCLUDE list (the scrubber first) **and** a build-time verification
pass that exits nonzero on any surviving identifier. Negative-tested both ways.
Payload still 23 MB. **Do not re-add the scrubber to the payload.**

### `8f91e32` The §4.1 validity audit — closes §3 item 1

Run `python3 scripts/emobank_validity.py` (~75 min, analysis-only, no GPU) →
`results/emobank_validity.json`. Design and rejected alternatives in
`scripts/_emobank_validity_design_notes.md`.

**Answer to the rebuttal-forcing question: the margin survives.** Against uncapped
TF-IDF *plus* sentence length and probe-vector scale, document-grouped:
**+0.103 / +0.094 / +0.085**, every interval clear of zero, positive in all five
document groups. Table 1's Supported row stands.

**Two things the paper had wrong, both now fixed in the .tex:**

1. **The folds leaked.** 10,062 sentences, **136 documents**, median 30 each,
   largest 11.8%. Shuffled folds put one document on both sides of a split, which
   inflates the *lexical baseline* far more than the probe. Grouping by document
   drops TF-IDF 0.303 → 0.215 on valence; the probe moves only 0.377 → 0.366. The
   published +0.074 was an **under**-statement, not an over-statement.

2. **§4.1 and §§4.2–4.6 do not measure the same quantity.**
   `run_emobank_validation.py:136` divides by the activation norm;
   `phase2_steering.py:212`, the writer for the trial files, does not. EmoBank
   probes are **cosines**; `V_int` is `‖ā‖cosθ`. Confirmed, not inferred: **1,719
   of 6,000** trial values exceed |1|. Note `src/experiments.py:82` *does*
   normalize and looks like the trial path — **it is not the writer for these
   files.** Check the writer, not the plausible-looking function.
   Under the direction-only readout (the only commensurable form) the margin
   narrows to +0.043/+0.054/+0.064 and **only dominance** stays clear of zero.
   Table 1 carries a ¶ footnote saying so.

**Also new:** convergent — "low-arousal negative" holds in *human* coordinates
(survivors average r = −0.092 V, −0.106 A vs −0.015/+0.012 for the other 42;
`grateful` and `nostalgic` are the hedged exceptions). Ecological — EmoBank median
70 chars vs the trials' 1,272, and **no** EmoBank sentence is inside the trials'
p5–p95 band; geometry transports (cos = 0.984) but the operating point does not
(`desperate` 1.68 SD out, 15.8% inside EmoBank's central 90%).

**Do not re-raise:**
- *"Compare raw EmoBank probes to raw trial probes."* Already tried; it gives a
  spurious "15 SD shift" because a bounded cosine is being compared to an
  unbounded projection. Only the unit-normalized readout is commensurable.
- *"Is the pipeline comparable to the published numbers?"* Yes — it reproduces all
  nine published EmoBank values to **six decimals**. Gate run before trusting any
  new number.
- *The survivor list.* Read it from
  `conditional_null.json → direction_only.overlap_with_scalar`. Typing it from
  memory produced a list wrong on 4 of 8.

**Gate is now 455 assertions** (was 400). Body still ends on page 5; the PDF is 12
pages because the appendix grew, and the limit excludes references and appendices.
