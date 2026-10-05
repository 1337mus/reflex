# Targeted training data

**Status: data ready.** Training has not run, so there are no model results.
The study compares old practice (control) with new practice (treatment). In the
treatment, HANS and WinoGrande replace one quarter of the old practice. See the
[study protocol](targeted-training-protocol.md) and
[audit summary](verification/targeted-training-data-summary.json).

| Task | New practice questions | Reserved test questions |
| --- | ---: | ---: |
| Sentence logic (HANS) | 600 | 300 |
| Commonsense (WinoGrande) | 600 | 200 |

Selection checks exact normalized matches across HANS premises and sentence
pairs, and WinoGrande IDs, sentences, option pairs, and conflicting answers. New
practice was checked against every approved evaluation question and ID. The audit
summary pins the sources, selection, schedules, test panel, bundle, and independent
audit without listing selected question IDs or gold answers.

| Execution check | Planned amount |
| --- | ---: |
| Training examples | 4,800 for each trained copy |
| Test questions, including changed answer order | 6,182 for each model version |
| Model calls | 28,210 total |
| Input text | 4,548,232 tokens total |
| Longest request | 631 tokens (2,048-token limit) |

Tokens are small pieces of text. The old-practice copy has 815,138 training tokens;
the new-practice copy has 693,016. Each copy sees the same number of examples. The
examples contain different amounts of text.

HANS training and test share all 60 templates, so this test does not cover unseen
templates. Exact checks can miss paraphrases, and the base model's exposure to
these sources during pretraining is unknown. The new practice combines both tasks,
so their separate effects cannot be measured. No training run or result is
available yet.
