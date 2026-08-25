# Handoff — 2026-08-25

Supersedes 2026-08-24. Everything below is committed and **pushed** on
`validity-fixes-2026-08-22`.

**Deadline: 2026-08-28 AoE — three days.** NeurIPS 2026 workshop
*Interpretability as a Science*, 5-page short paper, non-archival, double-blind,
OpenReview at `NeurIPS.cc/2026/Workshop/InterpScience`.

---

## 1. State

Gate **467/467**. Build clean, 12 pages, body ends p5, refs start p6.
`paper_numbers.py` regenerates `results/paper_numbers.json` byte-identical from
raw results. Anonymous zip builds, passes its own leak check, and now also
excludes `docs/writing_style.md`.

Today's commits:

    0e01120  Write the paper the way accepted NeurIPS workshop papers are written
    829a1b4  Repoint a gate check that passed by never running

---

## 2. What is left before submitting

1. **Reciprocal reviewer signup** — a separate Google form, NOT part of the
   OpenReview flow: https://forms.gle/DCUMr9WMWwn3pN6FA . The CFP requires at
   least one author. Easy to forget; nothing in the submission flow prompts you.
2. **Upload** the PDF plus `supplementary.zip` (build with
   `sh scripts/make_anonymous_zip.sh <outdir>`).
3. *Optional, 2 min:* cite the target's arXiv version, `arXiv:2604.07729`,
   alongside the Transformer Circuits URL in `refs.bib`.

That is the whole list.

---

## 3. The writing pass — read `docs/writing_style.md` before touching prose

The paper was rewritten to match a measured corpus of **121 papers whose arXiv
comment field states acceptance at a NeurIPS workshop** (23 PDFs, 14 LaTeX
sources analysed). Every rule and every number is in `docs/writing_style.md`,
including the corpus ids, so nothing has to be taken on trust.

Headline changes:

- **Abstract rewritten.** Was 257 words, 16 sentences, **17 result numbers**,
  opening on a citation and closing on a data-availability note. Now 199 words,
  9 sentences, **zero result numbers**, opening on the field and closing on an
  implication. Corpus median is 151 words in 6 sentences, and 53% of accepted
  abstracts carry no result number.
- **All 11 section headings are now noun phrases.** Four were claim-sentences.
  Corpus rate for verb-containing numbered headings is 2% (7 of 334 in source).
- **No semicolon, colon or interrupting dash left in body prose.** Ranges, name
  pairs, one table cell and one caption keep theirs.
- **Acronyms expanded**: LLM, AUC, TF-IDF. PCA dropped in favour of words.

**Two owner decisions, both taken 2026-08-25:**

- *The dead gate check* → **repointed** (see §4).
- *The text-baseline finding (old abstract item iv)* → **left out of the
  abstract.** It stays in Table 1 on page 1 and in its own subsection. Reason:
  the abstract is already above corpus median length and Table 1 is on page 1.
  Do not re-add without a reason to spend the ~20 words.

---

## 4. Do not re-raise — settled with evidence

Everything in the 08-22 and 08-24 handoffs' "settled" lists still stands
*except the two entries corrected in §5*. Added today:

- **The gate had a check that passed by never running.** It was keyed on the
  phrase "Four further standard controls" plus a count of `(i)-(v)` items. The
  rewrite deleted both, so its condition could never be true, yet it still
  counted toward the 467 because the counter ticks before the test. It now
  asserts that the abstract carries **no result numbers**. Verified by
  injecting `AUC $0.832$` into the abstract, confirming the gate reports it,
  and confirming green on removal. **A check that has never been seen to fail
  is not a check.** Count is still 467, so the figure the paper states is
  unchanged.
- **`\paragraph` headings are deliberately NOT noun phrases.** The 2% rule is
  for numbered headings. Run-in paragraph headings carry a finite verb in 9% of
  the corpus (7 of 75), with real examples like "The Pythia gap is an
  SAE-utilisation artefact." Ours sit at 36%, above the corpus but inside its
  practice. Do not "fix" them to match the section rule.
- **Our body punctuation is stricter than the entire corpus.** Colons per 1000
  words: corpus minimum 0.23, ours 0.00. This is a deliberate house rule, not
  an oversight, and not evidence of a bug.

---

## 5. Two inherited claims that did NOT survive checking

Both came from the 08-24 handoff's "settled with evidence" list. Both were
wrong. This is why that list is not self-certifying.

- **"The abstract length is not fixable — do not try again."** It was fixable.
  257 words to 199, and the enumerated list that was said to be load-bearing
  was removed without loss; every one of the 9 claims in the new abstract still
  traces to body text that says it. The gate did once reject a cut, but that
  rejection was about a phrase it pinned, not about the abstract's structure —
  and those pinned phrases were simply re-homed into the body.
- **"Hedge density is 3.09/1k vs a peer range of 0.00–0.35."** Measured against
  the 121-paper corpus on a standard hedge list: corpus median **3.42 per 1000
  words**, range 0.29 to 10.02; **ours is 0.31**, second-lowest of the set. The
  earlier list is unknown and hedge counts are list-dependent, but a peer range
  topping out at 0.35 is not credible for any reasonable list. The advice that
  followed it — do not strip the hedges — still holds, for the opposite reason:
  there is no surplus to cut.
- **The six arXiv ids the 08-24 handoff cited are real.** 2509.19943,
  2509.23717, 2510.03282, 2511.00059, 2511.06739, 2511.08854 all exist and all
  state acceptance at the NeurIPS 2025 Workshop on Mechanistic
  Interpretability. That claim checked out; it just left no artifact in the
  repo, which is why it had to be redone from scratch.

---

## 6. The process lesson, updated

The 08-22 lesson (prefer one cold read over another self-review) and the 08-24
lesson (the gate protects the `.tex` and nothing else) both still hold. Today
adds two:

**A passing check and a working check are different things.** The enumerated-
items check was green for two days while testing nothing. If you add a gate
assertion, break the thing it guards once and watch it fail before you trust it.

**A "settled with evidence" list decays.** Two of its entries were wrong when
re-measured, and one of them ("not fixable, do not try again") was actively
steering future work away from a real improvement. Date the entries, record how
each was measured, and re-measure before citing one as a reason not to act.

---

## 7. Open

1. **Merge to `main`?** Still unmerged. `main` is at the 08-21 handoff.
2. Parent repo `/home/gamir/naamarozen/gfs` has unrelated modified files under
   `results/final_run_v1/` and `taskB_logs/` belonging to the 70B false-premise
   sweep. Left alone deliberately.

---

## 8. How to verify anything

    python3 scripts/paper_numbers.py            # every number -> results/paper_numbers.json
    python3 scripts/h5_holdout_merged.py        # the registered rule on merged n=120
    python3 scripts/conditional_null.py         # family-wise + per-direction nulls (~30 min)
    python3 scripts/emobank_validity.py         # section 4.1 validity audit (~75 min)
    python3 scripts/check_paper_consistency.py  # 467 assertions gating the .tex
    sh      scripts/make_anonymous_zip.sh /tmp/anon   # supplementary + leak check

All analysis-only, no GPU, no network, seeded 20260819. **Re-run the gate after
any paper edit.**

Build: `/home/gamir/naamarozen/bin/tectonic -X compile paper/interpscience_short.tex`
