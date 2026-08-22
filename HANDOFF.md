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

1. **The rebuttal-forcing question, unanswered.** §5 tells the field to carry a
   simple output baseline. §4.1 — the one **Supported** row — carries only TF-IDF.
   Does the +0.074 / +0.083 / +0.087 margin survive a *length-and-magnitude-only*
   readout on EmoBank under the same folds and estimator? Appendix D re-ran §4.3
   under the magnitude-removed readout but never §4.1. If the margin doesn't
   survive, Table 1 has zero supported rows and the paper changes character. This
   is a new analysis, not an edit. It is the single most exploitable gap.
2. **Merge to `main`?** Branch is pushed; merge is a one-liner when you want it.
3. **Artifact:** anonymized zip vs Anonymous GitHub. Check the OpenReview form has a
   supplementary field. If you use the zip, consider dropping `HANDOFF.md` and
   `CONTINUATION_NOTES.md` — reviewers don't need working notes.
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
