# Design note: what §4.1's one SUPPORTED verdict actually rests on

## What must be tested

Table 1 (`interpscience_short.tex:105`) carries one \textsc{supported} row —
"Emotion subspace carries affect", control "EmoBank vs. TF-IDF". §5 (`:415`)
tells the field that *"Probe evaluations should carry a simple output baseline."*
The one row we support does not carry one. That asymmetry is the paper's most
exploitable gap and it is a **validity** gap, so it is worked through the
validity taxonomy rather than patched with one extra number.

Everything below is analysis-only: `results/emobank/llama70b/trials.jsonl` stores
`text`, human `V/A/D`, the 50 projections, and a document-bearing `id`. No GPU,
no re-generation.

---

## Construct validity — is the probe measuring affect, or activation size?

`V_int[e] = <ā, u_e>` is a scalar projection, so it scales with ‖ā‖.
`appendix_readout` establishes that normalizing the 50-vector to unit length
cancels that shared factor *exactly*, and that doing so changes which directions
survive in §4.3 (only 8 of 18). §4.1 was never re-run under that readout.

**Test.** Re-run §4.1 with each trial's 50-vector L2-normalized. If R² and the
margin survive, magnitude is not doing the work. If they collapse, the
"subspace carries affect" verdict was reading activation size.

**Caveat, stated rather than hidden:** ‖Pā‖ = ‖ā‖·‖Pâ‖, so the norm of the
50-vector is a magnitude proxy contaminated by direction. ‖ā‖ itself was never
stored (`appendix_readout`). Normalization removes the shared factor exactly;
the *feature* named `probe_norm` below is a proxy, and is labelled as one.

## Content validity — does the baseline cover the space of trivial explanations?

A length-and-magnitude baseline covers surface size. It does not cover lexical
content; TF-IDF does. Neither alone is the content-valid baseline.

**Test.** Four feature sets under one estimator and one fold structure:
`trivial` = [n_chars, n_words, ‖50-vector‖]; `tfidf` (uncapped, the existing
row); `tfidf+trivial` = the union, which is the honest strongest baseline; and
the probe. The file's own stated principle — *"A baseline is treated
conservatively by being given more capacity, not less"* — requires the union row
to be the one the verdict is read from.

Content validity of the *construct* side is also uneven and already visible:
V 0.377 ≫ A 0.180 > D 0.154. The 50 labels cover valence far better than
arousal or dominance. Report whether the *margin* is uniform even though the
*level* is not.

## Discriminant validity — does the probe fail to reduce to the nuisance?

**Test.** R² of `trivial` alone. If length and magnitude alone approach the
probe's 0.377 on valence, the probe does not discriminate from the nuisance and
the row falls regardless of what it does against TF-IDF.

**Limit, stated:** the sharpest discriminant control — 50 *random* directions on
the same activations, as in `random_subspace_null.py` for §4.3 — is not
available here. Only the 50 emotion projections were stored for EmoBank, not the
activations. This cannot be run without GPU and is declared as a gap, not
quietly skipped.

## Convergent validity — does the probe agree with an independent measure?

§4.3 and `appendix_readout` describe the surviving axis as "low-arousal
negative". That label is currently read off the *emotion words* — off what
`bored` and `gloomy` mean in English — not off data.

**Test.** Correlate each of the 50 single projections with EmoBank's human V, A
and D. This puts every direction at a coordinate that humans, not the authors,
assigned. If the §4.3 survivors really sit low-arousal negative, they land there
in human coordinates. If they do not, a descriptive claim in the paper is
unsupported and must be reworded.

## Criterion validity — concurrent and predictive

Concurrent criterion = human V/A/D (this analysis). Predictive criterion =
reward hacking (§4.2–4.3, where the probe fails once length is controlled).
The paper already reports the predictive failure. Nothing new here; recorded so
the taxonomy is not silently truncated to the flattering half.

## Internal validity — does the fold structure license the estimate?

**This is the threat the existing analysis does not control.** The 10,062
sentences come from **136 documents** (median 30 sentences per document, one
with 1,192). `KFold(5, shuffle=True)` therefore places sentences from the same
document in train and test in effectively every fold. Both rows are inflated,
but not equally: TF-IDF can memorize document vocabulary, and the probe can ride
document-level activation style. The *direction* of the bias on the **margin**
is not determinable a priori — which is exactly why it has to be measured.

**Test.** Repeat every contrast under `GroupKFold(5)` grouped on document id, so
no document spans the split. Both readings of the margin — random-fold and
grouped — are reported. If they disagree, the grouped one is the estimate,
because the random one leaks.

## External validity — does it hold beyond one slice of the corpus?

**Test.** Recompute the grouped margin within held-out document groups and check
whether it is positive across them rather than driven by one large document
(the 1,192-sentence one is 12% of the corpus on its own).

## Ecological validity — is EmoBank the setting the probe is used in?

The probe earns its construct-validity evidence on **third-party corpus
sentences read as input**, then is applied in §4.2–4.5 to **model-generated
chain-of-thought during a reward-hacking task**. Different author, different
position in the sequence, different length regime.

**Test.** Compare EmoBank sentence length against the 120 trial response lengths,
and the distribution of `V_int[desperate]` in each. If the two barely overlap,
§4.1's evidence is being transported across a domain gap, and that belongs in
the paper as a limitation with a number attached rather than as a vague hedge.

---

## What can change in the paper

A verdict flip is permitted in one direction only: if the margin does not
survive, Table 1's one SUPPORTED row weakens and the body says so. Numbers that
survive are added; numbers already in the paper are not touched. Existing keys
in `results/emobank_baseline.json` are appended to, never rewritten, so the
400-assertion gate cannot silently pass on changed values.

Seeds: fold seed 20260510 throughout, matching `run_emobank_validation.py`.

---

# Addendum, same day: reading the two writers changed the question

The construct-validity section above assumed §4.1 and §4.2–4.5 measure the same
quantity and differ only in domain. **They do not.** Established by reading the
code that writes each file, not by inference:

| | §4.1 EmoBank | §4.2–4.5 trials |
|---|---|---|
| writer | `run_emobank_validation.py:136` | `phase2_steering.py:212` |
| formula | `dot(r,v) / (‖r‖·‖v‖)` | `dot(ā,v) / ‖v‖` |
| quantity | **true cosine** | **scalar projection** |
| position | **last token** | **mean from token 50 on** |
| format | raw sentence, no chat template | chat template |

`src/experiments.py:82` does normalize and looks like the trial path, but it is
not the writer for these files. Checking which function actually produced the
data mattered; the plausible-looking one was the wrong one.

**Empirical confirmation, not just code reading:** 1,719 of 6,000 trial values
exceed |1|, which no cosine can. Every EmoBank value lies inside [−1, 1].

## Three consequences

1. **The original question is partly moot.** "Does §4.1's margin survive a
   magnitude-removed readout?" — §4.1 was *already* magnitude-free. ‖r‖ is
   divided out at line 136. The scalar readout that the behavioral sections run
   on was never applied to EmoBank at all, and cannot be reconstructed: ‖r‖ was
   not stored, exactly as `appendix_readout` says of ‖ā‖.

2. **`probe_unit` must be re-described.** On EmoBank it does not remove
   activation magnitude — that is already gone. It removes the remaining overall
   scale of the 50-vector. It is still the right feature set to run, for the
   reason in (3), but the label "magnitude-removed" would be wrong here.

3. **It is the one commensurable readout, which is why it matters.**
   EmoBank `c/‖c‖ = P r/‖P r‖`; trials `p/‖p‖ = P ā/‖P ā‖`. Identical functional
   form, each cancelling its own scale factor exactly. Unit-normalizing is the
   only readout in which §4.1 and §4.3 are the same measurement, so every
   probe-geometry comparison between the two settings is made there.

## An error this caught in my own first pass

Before reading the writers I compared raw EmoBank values to raw trial values and
found a "15 SD shift" with completely disjoint 50-vector norms (0.58 vs 6.26,
10.8×). That is a **readout artifact** — a bounded cosine against an unbounded
projection — and is not reported anywhere. The length comparison from the same
pass involves no readout and stands.

## What survives from the original plan

Construct, content, discriminant, convergent, criterion, internal and external
are unaffected: they compare feature sets *within* EmoBank, under one readout,
one fold structure and one estimator. Ecological is the section that had to be
rebuilt, and it is now stronger: the readout mismatch is itself an ecological
finding with file-and-line provenance, not a hedge.

**Comparability gate, run before any of this was trusted:** the new pipeline
reproduces all nine published EmoBank values — probe R², TF-IDF R² and the three
paired margins — to six decimal places. Any new number from it is therefore
directly comparable to what is already in the paper.
