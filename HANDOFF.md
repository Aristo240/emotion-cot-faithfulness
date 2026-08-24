# Handoff — 2026-08-24

Supersedes the 2026-08-22 / 08-23 version. Everything below is committed on
`validity-fixes-2026-08-22`. Working tree clean. **Not pushed** — see §6.

**Deadline: 2026-08-28 AoE — four days.** NeurIPS 2026 workshop *Interpretability
as a Science*, 5-page short paper, non-archival, double-blind, OpenReview at
`NeurIPS.cc/2026/Workshop/InterpScience`.

---

## 1. State

Gate **467/467**. Build clean, 12 pages, body ends p5, refs start p6.
`paper_numbers.py` regenerates `results/paper_numbers.json` byte-identical from
raw results (re-verified 08-24). Anonymous zip builds, 24 MB, passes its own
leak check, zero junk files.

Three commits today:

    996f325  Keep the abandoned human-validation worksheet out of the payload
    f283a09  Back the release claim; drop the defensive novelty framing
    0f99742  The supplementary contradicted the paper on the registered verdict

---

## 2. What is left before submitting

1. **Reciprocal reviewer signup** — a separate Google form, NOT part of the
   OpenReview flow: https://forms.gle/DCUMr9WMWwn3pN6FA . The CFP requires at
   least one author. Easy to forget; nothing in the submission flow prompts you.
2. **Upload** the PDF plus `supplementary.zip` (build with
   `sh scripts/make_anonymous_zip.sh <outdir>`).
3. *Optional, 2 min:* cite the target's arXiv version, `arXiv:2604.07729`,
   alongside the Transformer Circuits URL in `refs.bib`.
4. *Optional:* hedge the abstract's "leaves 18 significant" with "(11 under
   every nuisance model)". §4.3 already says the count is the fragile part;
   the abstract quotes only the favourable number.

That is the whole list. Under an hour of work.

---

## 3. Verdict on readiness

**Ready.** Estimated acceptance **75–80%**, poster rather than contributed talk.

Venue fit is the strongest single factor and was under-appreciated until 08-24:

- **Four of the seven organizers** (Joshi, Klindt, Reizinger, Sridhar) wrote
  `joshi2026causality`, which is the paper's *opening citation* and frames the
  whole thesis. Verified against `refs.bib` and the workshop site.
- The CFP's topic list names "Measurement validity, identifiability, and
  evaluation design" and "Falsifiability and experimental designs that
  distinguish mechanisms from artifacts." Two Table 1 verdicts are literally
  *Not identifiable*; §4.2 is a mechanism-vs-artifact separation.
- CFP confirms **double-blind** (the .tex header comments still say the policy
  is unstated — stale, harmless, source is not submitted).

What a reviewer will push on, all disclosed in the paper already: power (14
events, four prompts, 1/39 in the key arm); conceded method novelty, with
Fomin et al. prior and Gözükara's successful Llama-3 steering left unexplained;
the one Supported row weakening to dominance-only under the commensurable
readout.

---

## 4. Do not re-raise — settled with evidence

Everything in the 08-22 handoff's §2 still stands. Added today:

- **The registered verdict is SUPPORTED, not INSUFFICIENT-DATA.**
  `min_tasks_required(4)` returns 3 (`h5_holdout.py:77-81`, hardcoded branch;
  `ceil(2*4/3)=3` agrees). 3 of 4 variants produce events. Re-running
  `h5_holdout_merged.py` prints `DECISION: SUPPORTED`, byte-identical to the
  stored JSON. The prereg said otherwise until today; it was wrong, and the
  correction is logged as dated deviations 5 and 6 rather than silently applied.
- **0.069 is dead.** It is exactly `chi2.sf(3.311045127775209, 1)` — verified to
  16 digits — an invalid reference distribution for a ridge-penalised statistic.
  Use T = 3.31, conditional-null p = **0.060**, family-wise max-T p = **0.440**.
- **EmoBank numbers are the `grouped` ones.** `cv_r2.grouped` (probe 0.366,
  union baseline 0.262) — *not* `cv_r2.random` (0.377 / 0.303), which are the
  leaked shuffled folds. If you see 0.377 anywhere, it is stale.
- **Human annotation was deliberately skipped.** Worksheet, blinded key and
  scorer are built and unused; `annotation/` is excluded from the zip. The
  defences that replace it: cross-judge 120/120 no abstentions, and Appendix F's
  adversarial flip analysis (`bored` T 38.7 → 24.5 → 16.2 vs a 7.9–8.5
  boundary). Do not re-litigate; if you want it, it is ~2–3 h of labelling and
  `scripts/score_human_validation.py` handles the rest.
- **Structure was checked empirically**, not assumed, against six accepted
  MechInterp NeurIPS 2025 papers (2509.19943, 2509.23717, 2510.03282,
  2511.00059, 2511.06739, 2511.08854). Pages 12 vs 9–22 and body/appendix 50%
  vs 33–67% are mid-distribution. Abstract is the longest (262 vs median ~177)
  and the only one with enumerated items; hedge density is 3.09/1k vs 0.00–0.35.
  **The abstract length is not fixable** — the gate rejected the one substantive
  cut because that list maps the abstract onto §4.2/4.3/4.5/4.4. Six words of
  fat existed and were removed. Do not try again.
- **Hedge density is genre-appropriate.** "Not identifiable" is verdict
  vocabulary, not throat-clearing. Do not strip it.

---

## 5. The process lesson, updated

The 08-22 lesson ("prefer one cold read over another self-review") still holds.
Today added a sharper one:

**The gate protects the `.tex` and nothing else.** `check_paper_consistency.py`
never reads `README.md` or `docs/preregistration.md`. That is exactly why they
drifted into contradicting the paper for two days while the manuscript stayed
correct through three audit rounds. When you correct a number, grep the whole
repo for the old value, not just the paper.

Extending the gate to cover those two files is possible but changes the
assertion count, which §3 of the paper states and the gate self-checks. Not
worth it four days out.

Second: **the gate caught a bad edit today.** It rejected removal of the
abstract's four-step list. Trust it over your own sense of what is padding.

---

## 6. Open

1. **Push?** `git push origin validity-fixes-2026-08-22`. Not done today.
2. **Merge to `main`?** One-liner when you want it; still unmerged.
3. Parent repo `/home/gamir/naamarozen/gfs` has unrelated modified files under
   `results/final_run_v1/` and `taskB_logs/` belonging to the 70B false-premise
   sweep. Left alone deliberately.

---

## 7. How to verify anything

    python3 scripts/paper_numbers.py            # every number -> results/paper_numbers.json
    python3 scripts/h5_holdout_merged.py        # the registered rule on merged n=120
    python3 scripts/conditional_null.py         # family-wise + per-direction nulls (~30 min)
    python3 scripts/emobank_validity.py         # section 4.1 validity audit (~75 min)
    python3 scripts/check_paper_consistency.py  # 467 assertions gating the .tex
    sh      scripts/make_anonymous_zip.sh /tmp/anon   # supplementary + leak check

All analysis-only, no GPU, no network, seeded 20260819. **Re-run the gate after
any paper edit.**

Build: `/home/gamir/naamarozen/bin/tectonic -X compile paper/interpscience_short.tex`
