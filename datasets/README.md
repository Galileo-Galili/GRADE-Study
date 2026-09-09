# Datasets — ICISER 2026 short paper

Three corpora, all audited. Every count below is measured, not taken from
the source's own description. Audit scripts are in `../scripts/audit/`.

| Dir | Role | Rows | Human:AI | License |
|-----|------|------|----------|---------|
| `llm_detect/` | train / val / test | 29,145 raw → **27,226 unique** | 16,042 : 11,184 | Kaggle competition terms |
| `hf_pile/` | held-out shift set | 20,000 (sampled) | 10,000 : 10,000 | MIT |
| `abstract_paraphrase/` | held-out probe | 304 | 152 : 152 | see `SOURCE_README.md` |

---

## llm_detect/ — training corpus

**`Training_Essay_Data.csv`** — columns `text`, `generated` (0=human, 1=AI).

Real student essays from the Vanderbilt / Learning Agency Lab
"LLM - Detect AI Generated Text" competition. Student spelling and
grammar are preserved. Two assigned prompts only:

1. Car-free cities (explanatory essay on limiting car usage)
2. Does the electoral college work? (letter to a state senator)

**`train_prompts.csv`** — the two prompt texts and their source passages.

### Audit findings

- **1,919 duplicate texts** (29,145 rows → 27,226 unique). Labels never
  conflict, so these are clean re-uploads. **Must be removed before
  splitting** or the same essay lands in train and test.
- 4 rows under 100 characters; one is a single character. Needs a
  minimum-length filter.
- No `prompt_id` column in this file. Prompt labels are recoverable by
  matching against the two prompts, needed only if cross-prompt
  evaluation is run.
- AI half generated with PaLM 2 (text-bison, text-unicorn), Gemini 1.0
  Pro, Gemini 1.0 Ultra and GPT-4; 3-shot prompted on same-topic student
  essays; deduplicated at 85% embedding cosine similarity; then
  distribution-matched to student essays on word count, sentence length
  and function-word count. No per-row generator label, so **no
  cross-generator claim is possible.**

---

## hf_pile/ — held-out shift set

**`hf_pile_sample_20k.csv`** — columns `source` (human/ai), `id`,
`shard`, `text`.

Balanced 10k/10k sample of `artem9k/ai-text-detection-pile` (1.39M rows,
MIT). Drawn with seed 42 across shuffled row groups; see
`../scripts/audit/hf_sample.py`. Shards are sorted by class, so human
rows come from shard 0 and AI rows from shard 6.

Human side: web/essay sources (IvyPanda essays, WebText, Reddit).
AI side: GPT-2, GPT-3, GPT-J, ChatGPT.

### Audit findings

Passes the checks that disqualified other corpora:

- No marker leakage; no `[AI-Generated]`-style giveaway.
- **Zero** cross-class duplicates.
- 10,000/10,000 human distinct; 9,974/10,000 AI distinct.
- 705,372 distinct human sentences — no template collapse.

**⚠ Length confound.** Median human document 7,323 chars vs AI 1,119 —
roughly 6:1. A logistic classifier on document length *alone* reaches
**81.4%**; TF-IDF reaches 97.9%. Truncation to 256 tokens removes part
of this. **The length-only baseline must be reported beside any detector
result on this corpus.**

---

## abstract_paraphrase/ — held-out probe

**`abstract_paraphrase_en.csv`** — columns `pair_id`, `source`
(Native/GPT), `label`, `filename`, `text`. Built by
`../scripts/audit/mk_abstract_csv.py` from `EnglishDatasetSample/`,
which is retained as the source of record.

English PhD-thesis abstracts (Belgrade / Singidunum), each paired with a
GPT-4 paraphrase generated at temperature 0.

### Audit findings

- 152 pairs, all complete: every `pair_id` has exactly one Native and
  one GPT member.
- Text length: min 782, median 2,639, max 24,615 chars.
- **⚠ `pair_id` is the required grouping key.** Each AI text is a
  paraphrase of its own human partner, so splitting a pair across
  train/test would put near-identical text on both sides. This corpus is
  inference-only here, so the constraint matters only if it is ever
  trained on.
- **⚠ Underpowered.** At 304 items one misclassification moves accuracy
  ~0.7 points. Report as a direction, not a result.

---

## Rejected corpora

Four candidates were audited and rejected. See §5.1 and Table 1 of the
paper, and `../audit_evidence/`.

| Corpus | Rows | Why rejected |
|--------|------|--------------|
| AI_vs_Human_Text_Dataset_v2 | 15,000 | Both classes drawn from the *same* 300 template sentences; 0 class-exclusive sentences; TF-IDF 51.4% vs 50% chance |
| my_dataset_expanded | 10,240 | Literal `[AI-Generated]` prefix on 100% of AI docs, 0% of human; 2,455 distinct sentences total; 232 texts in *both* classes with contradictory labels |
| AIDE (tla-lab) | 1,378 | 100% subset of `llm_detect`; only 3 AI rows |
| yashdogra/nlpdataset | 690 | Off-task — news sentence classification, not AI-vs-human |
