# Pre-registered write-up decisions

Rules fixed **before** the results existed, so that a decision about what goes
in the paper cannot be made by looking at which version reads better. Dated
2 September 2026.

---

## The abstract-corpus side probe

**What it is.** Each fold of the leave-one-assignment-out sweep also scores the
Abstract Paraphrasing corpus: 304 PhD-thesis abstracts, 152 human and 152
paraphrased by GPT-4. Results land in `results/loto_abstract_probe.csv`, one
row per fold and seed.

**Why it is tempting.** Qorich and El Ouazzani report **97.38%** on this exact
corpus, at this exact 152:152 split, under five-fold cross-validation — which
trains on four fifths of the abstracts before testing on the rest. Scoring the
same corpus without training on it at all is a direct comparison on their own
data.

**Why it is not a research question here.** It changes two things at once
relative to training: the register (school essay to PhD abstract) *and* how the
generated text was produced (composed from a prompt to reworded from a human
original). It is therefore not a controlled comparison. The same objection
removed the AI Text Pile and the GLC paraphrase corpora from this study. At 304
items it is also underpowered.

### The rule

If it goes in at all, it goes in as **one sentence in the related-work
section**, supporting the argument that cross-validation cannot measure
transfer. Not a research question, not a table, not a figure, not a subsection.

**Include it only if all four hold:**

1. **Consistent across folds.** The spread in accuracy across the 15 folds is
   **at most 15 percentage points**. A wider spread means the number is
   fold-dependent noise and says nothing about the corpus.
2. **A real contrast.** Mean accuracy is **below 90%**. At or above that there
   is no meaningful gap from 97.38% and the sentence makes no point.
3. **Not a single-class collapse presented as a score.** If recall sits at or
   near 100% while the false-positive rate also sits near 100%, the detector
   has stopped discriminating rather than scored poorly. That may still be
   reported, but it must be described as a collapse, with both rates given. It
   must never be written as though it were an accuracy figure.
4. **Honest about the confound.** The sentence must state that register and
   generating task both differ from training, so the number bounds transfer
   rather than explaining it.

**Exclude it entirely if:**

- the spread across folds exceeds 15 points, **or**
- mean accuracy is 90% or above, **or**
- it cannot be stated in one sentence without needing a table to interpret.

Excluding it costs the paper nothing. The two research questions do not depend
on it, and the related-work argument about cross-validation already stands on
the published method alone.

### If excluded

Leave `EVAL_ABSTRACTS = True` in `scripts/train_loto.py`. The probe stays in
the repository as a recorded negative, so the decision is checkable. It simply
is not mentioned in the paper.

---

## Standing rules carried forward

**No invented numbers.** Any quantity not yet measured appears as a visible
`--` in tables and is never written as a plausible-looking figure.

**Controls per corpus.** Every evaluation set gets the length-only and
word-frequency comparisons fitted on that set. A detector that fails to clear
them has demonstrated nothing there.

**Audits raise, not warn.** The split audit halts the study on any leak. It is
never downgraded to a printed warning.

**Recovered labels are not trusted.** Assignment labels come from the corpus
that records them. The earlier recovery, which assumed two assignments and got
24% of one of them right, is discarded and is not reinstated.
