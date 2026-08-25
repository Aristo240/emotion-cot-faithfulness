# Handoff — 2026-08-25 (evening)

Supersedes 2026-08-24. Everything below is committed and **pushed** on
`validity-fixes-2026-08-22`.

**Deadline: 2026-08-28 AoE — three days.** NeurIPS 2026 workshop
*Interpretability as a Science*, 5-page short paper, non-archival, double-blind,
OpenReview at `NeurIPS.cc/2026/Workshop/InterpScience`.

---

## 1. State

Gate **467/467**. Body **5 pages** (measured correctly, see §3d), 13 pages total,
references start p6. Anonymous zip builds and passes its own leak check.
`paper_numbers.py` regenerates `results/paper_numbers.json` byte-identical.

All work is committed and pushed on `validity-fixes-2026-08-22`. Nothing is
running in the background; there is no job to babysit.

## 2. What is left before submitting

**Deadline 2026-08-28 AoE.**

1. **Reciprocal reviewer signup** — a separate Google form, NOT part of the
   OpenReview flow: https://forms.gle/DCUMr9WMWwn3pN6FA . The CFP requires at
   least one author, and nothing in the submission flow prompts you.
2. **Upload** the PDF plus `supplementary.zip`
   (`sh scripts/make_anonymous_zip.sh <outdir>`).
3. *Optional, 2 min:* cite the target's arXiv version `arXiv:2604.07729`
   alongside the Transformer Circuits URL in `refs.bib`.
4. *Optional, the one substantive gap left.* \citet{gozukara2026emovecllm}
   replicate the target's extraction on open weights including Llama-3 and
   report successful steering. Related work now notes their readout is taken
   before generation and ours is not, and says we have not tested whether that
   accounts for the difference. Two or three sentences developing that would
   turn the paper's weakest point into a case its own framework explains. It is
   the highest-value edit remaining and it is not required.

**Do not start another restructuring or style pass.** Structure, title, ordering
and prose are settled. See §3b, §3c, §4.

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
- **No semicolon, colon or interrupting dash left in body prose or in any of
  the six appendices.** Ranges, name pairs, one table cell and one caption keep
  theirs. Note `appendix_directions.tex` is generated and byte-gated: edit
  `scripts/make_appendix_table.py`, do not hand-edit the output.
- **A cover-to-cover read after all of the above caught four more things**, three
  of which the automated audit had missed. See §5.
- **Acronyms expanded**: LLM, AUC, TF-IDF. PCA dropped in favour of words.

**Two owner decisions, both taken 2026-08-25:**

- *The dead gate check* → **repointed** (see §4).
- *The text-baseline finding (old abstract item iv)* → **left out of the
  abstract.** It stays in Table 1 on page 1 and in its own subsection. Reason:
  the abstract is already above corpus median length and Table 1 is on page 1.
  Do not re-add without a reason to spend the ~20 words.

---

## 3b. The reframe (2026-08-25, owner decision)

A reviewer read found the paper reading as a list of its own weaknesses, with the
one genuinely underpowered result in front. Reframed. **No number, no verdict and
no Table 1 row changed.** What changed is which claim carries the paper.

Title is now *Response Length Passes the Same Test as an Emotion Probe for Reward Hacking*. It must say **registered**, not preregistered, because §3 defines the term
as planned-not-pre-data and the gate checks it.

The argument for the reframe, in one line: the three strongest results do not
depend on sample size, and the old framing buried them.

1. A character count meets every threshold of the registered criterion. A
   demonstration, so one case is the whole argument.
2. The tie share of the text comparison is a counted property of the data.
3. A pre-generation probe takes one value per prompt, so it cannot be evaluated
   in a suite of many rollouts over few prompts. Deductive.

The steering arm, one event in 39 trials, is the least load-bearing thing in the
paper and is now presented as such. Full reasoning in `docs/writing_style.md`.

**Do not write that the character count "carries no information about affect."**
That was drafted and removed. §4.2 allows length to mediate a genuine effect, so
it contradicts the paper. The supportable claim is that a criterion a character
count clears cannot license a claim about affect.

## 3c. Rhetorical restructure — DONE, do not redo

Restructured against three NeurIPS 2025 MechInterp workshop papers read in full
(arXiv 2510.00845, 2509.03888, 2507.06445). A fourth requested comparator,
"Control and Predictivity in Neural Interpretability", could not be found on
arXiv, on the workshop site or by search, and was not used.

**Settled. Do not re-sort, re-title or re-litigate any of this.**

Results order, deliberate:
1. `sec:prereg` a character count satisfies the registered criterion (headline)
2. `sec:sem` the emotion subspace carries affect information
3. `sec:len` not separable from response length
4. `sec:dirs` selection correction
5. `sec:vtext` tie resolution
6. `sec:causal` steering

Why 2 sits where it does: it blocks the reading that the probes are noise, which
is what makes 1 non-trivial. It carries an explicit sentence saying it is **not**
evidence for the `desperate` direction and **not** for the unnormalized behavioral
readout. Never let its adjacency to 1 imply "we validated the probe, then length
confounded it".

Title: *Response Length Passes the Same Test as an Emotion Probe for Reward
Hacking*. Two earlier drafts were rejected. **No first person** (0 of 121 corpus
titles use it) and **no "registered"** (a term §3 must define before it means
anything).

In the appendix, moved there for the page limit and nothing else: the eight-claim
summary table, the steering-arms table, and the judge-agreement diagnostics. All
still inside `paper/interpscience_short.tex`, after `\appendix`. No
`appendix_*.tex` file was edited.

Fidelity audit vs the pre-restructure file: no numeric value appeared or
vanished (137 distinct tokens both sides), the citation set is identical at 20
keys, Table 1's verdict column is byte-identical, and all 21 scientific
qualifiers checked for still stand. A claim-number binding audit compared the
context sentence of 37 key values old vs new: **all 37 identical**, so no number
was reattached to a different population, analysis or null.

## 3d. The page limit was being measured wrong

**This matters more than anything else in this handoff.** For days this file
claimed "body ends p5, refs start p6". That was produced by finding the page the
References heading starts on and subtracting one. Body text and the References
heading can share a page, so the check silently passed while the body ran over.

Measured correctly, the body was **6 pages** from commit `0f00edd` onward and I
did not notice until the restructure pushed it to 635 words over. It is now back
to 5.

Correct check, use this one:

    pdftotext -layout paper/interpscience_short.pdf - | python3 -c "
    import sys,re
    pg=sys.stdin.read().split(chr(12))
    for i,p in enumerate(pg,1):
        q=re.sub(r'^ *\d{1,3}   ?','',p,flags=re.M)
        j=q.find('References')
        if j>=0:
            w=len(q[:j].split())
            print('body pages:', i if w>20 else i-1); break"

If it prints 6, the paper is over the limit however green the gate is. The gate
does not know about pages.

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

## 5. What the cover-to-cover read caught, and one audit that lied

Read the compiled PDF end to end after the style pass. Four fixes, and the
reason three of them were needed is worth more than the fixes:

- **The punctuation audit had a silent false-negative.** It stripped LaTeX
  comments with `line.split("%")[0]`, which also truncates at every escaped
  `\%`. The paper is full of `$95\%$`, so roughly a third of the prose was
  never examined and the audit reported "clean" while a semicolon and two
  colons sat in the body. Corrected method and the other two traps are in
  `docs/writing_style.md`. **Print the surviving word count of any such script;
  if it is far below the rendered body, its zero means nothing.**
- **"registered in advance" in the abstract contradicted §3**, which defines
  "registered" as planned-not-pre-data because the plan was written after an
  exploratory AUC of 0.900. Introduced by the rewrite, caught by reading. The
  gate's timing check only matches "preregistered"/"preregistration", so it
  could not see it. Now reads "registered", the paper's own defined term.
- **Three doubled connectives** left by the punctuation pass, where replacing a
  colon or semicolon with a conjunction produced "so ... so", "so ... because"
  and "and ... and". Splitting a sentence is usually better than conjoining it.
- The appendices were then audited with the corrected method and cleaned too,
  32 violations in five files.

## 6. Two inherited claims that did NOT survive checking

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

## 7. The process lesson, updated

The 08-22 lesson (prefer one cold read over another self-review) and the 08-24
lesson (the gate protects the `.tex` and nothing else) both still hold. Today
adds two:

**A passing check and a working check are different things.** The enumerated-
items check was green for two days while testing nothing. If you add a gate
assertion, break the thing it guards once and watch it fail before you trust it.

**An audit you wrote is evidence only after you have seen it fail.** The
punctuation audit reported the body clean while three violations sat in it,
because its comment-stripping quietly ate a third of the text. The same session
had just repointed a gate check for exactly this failure mode and still shipped
it in a throwaway script. Print the word count. Break the thing on purpose.

**A "settled with evidence" list decays.** Two of its entries were wrong when
re-measured, and one of them ("not fixable, do not try again") was actively
steering future work away from a real improvement. Date the entries, record how
each was measured, and re-measure before citing one as a reason not to act.

---

## 8. Open

1. **Merge to `main`?** Still unmerged. `main` is at the 08-21 handoff.
2. Parent repo `/home/gamir/naamarozen/gfs` has unrelated modified files under
   `results/final_run_v1/` and `taskB_logs/` belonging to the 70B false-premise
   sweep. Left alone deliberately.

---

## 9. How to verify anything

    python3 scripts/paper_numbers.py            # every number -> results/paper_numbers.json
    python3 scripts/h5_holdout_merged.py        # the registered rule on merged n=120
    python3 scripts/conditional_null.py         # family-wise + per-direction nulls (~30 min)
    python3 scripts/emobank_validity.py         # section 4.1 validity audit (~75 min)
    python3 scripts/check_paper_consistency.py  # 467 assertions gating the .tex
    sh      scripts/make_anonymous_zip.sh /tmp/anon   # supplementary + leak check

All analysis-only, no GPU, no network, seeded 20260819. **Re-run the gate after
any paper edit.**

Build: `/home/gamir/naamarozen/bin/tectonic -X compile paper/interpscience_short.tex`
