# BoolQ and SNLI broader baseline results

The frozen two-task development comparison completed on 2026-10-04. It scored
three pinned models on 32 BoolQ groups and 32 SNLI groups. Kev led SNLI and tied
Intern on BoolQ accuracy, with fewer observed option-order changes in this
sample. This remains a small development pilot: the intervals are wide, source
overlap is not fully known, and no Reflex-trained model was evaluated. It shows
no Reflex advantage or broad model ranking. SNLI accuracy and option-order
robustness remain comparative headroom to investigate, not established Reflex
improvements.

## Primary results

| Model | Dataset | Original order correct / 32 (95% Wilson interval) | Groups changing answer in any order / 32 | Macro-F1 |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B-Base | BoolQ | 18/32 (39.3–71.8%) | 18 | 0.5556 |
| Intern-Decision-0.8B | BoolQ | 26/32 (64.7–91.1%) | 4 | 0.7500 |
| Kev-0.8B | BoolQ | 26/32 (64.7–91.1%) | 2 | 0.7681 |
| Qwen3.5-0.8B-Base | SNLI | 13/32 (25.5–57.7%) | 32 | 0.3632 |
| Intern-Decision-0.8B | SNLI | 13/32 (25.5–57.7%) | 24 | 0.2900 |
| Kev-0.8B | SNLI | 22/32 (51.4–82.0%) | 2 | 0.6898 |

Wilson intervals treat the 32 distinct source groups per dataset as sampling
units. The BoolQ mirror lacks page titles, so distinct passage groups do not
establish page-level independence. Each group is counted as order-sensitive if its semantic answer changes in at least one
of the two BoolQ or six SNLI option permutations. Those permutations are paired
repeated measurements, not additional independent questions. No ties occurred
in the original order or across the measured permutations.

## Position behavior

Histograms count the position of the gold and predicted semantic answer in the
original option order: BoolQ `(yes, no)` and SNLI
`(entailment, neutral, contradiction)`. Intern predicted `neutral` for 30 of 32
SNLI groups; its remaining two predictions were `entailment`, with none for
`contradiction`.

| Model | BoolQ gold positions | BoolQ predicted positions | SNLI gold positions | SNLI predicted positions |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B-Base | 22, 10 | 14, 18 | 10, 11, 11 | 16, 15, 1 |
| Intern-Decision-0.8B | 22, 10 | 26, 6 | 10, 11, 11 | 2, 30, 0 |
| Kev-0.8B | 22, 10 | 24, 8 | 10, 11, 11 | 8, 19, 5 |

The position pattern and SNLI flips indicate substantial order sensitivity for
Qwen and Intern on this slice. Kev changed only two group decisions in either
task, but two small samples cannot establish general order invariance.

## Confidence metrics

Each cell reports `NLL / Brier / ECE`; lower is better. Raw values use
temperature `T=1`. The shipped column applies the pinned checkpoint or Hub
temperature. Intern's value is its Hub default; the repository preset is
XTuner-bound and was not used. Kev's value comes from pinned checkpoint
metadata. Qwen has no shipped calibration. ECE is the preregistered ten-bin
top-label estimate and is noisy at 32 examples. Temperature scaling did not
change top-1 accuracy.

| Model | Dataset | Raw (`T=1`) NLL / Brier / ECE | Shipped temperature | Shipped NLL / Brier / ECE |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B-Base | BoolQ | 0.6715 / 0.4795 / 0.1163 | 1.0 (none) | 0.6715 / 0.4795 / 0.1163 |
| Intern-Decision-0.8B | BoolQ | 0.4041 / 0.2519 / 0.1321 | 2.747760550703 | 0.4708 / 0.2973 / 0.1309 |
| Kev-0.8B | BoolQ | 0.3193 / 0.2179 / 0.1323 | 2.3510958125672174 | 0.3254 / 0.2011 / 0.1224 |
| Qwen3.5-0.8B-Base | SNLI | 1.0461 / 0.6372 / 0.1157 | 1.0 (none) | 1.0461 / 0.6372 / 0.1157 |
| Intern-Decision-0.8B | SNLI | 1.9495 / 0.9716 / 0.4629 | 2.747760550703 | 1.1140 / 0.7012 / 0.2375 |
| Kev-0.8B | SNLI | 1.1638 / 0.5242 / 0.2440 | 2.3510958125672174 | 0.7462 / 0.4219 / 0.1276 |

Macro-F1 uses the original-order predictions. Temperature did not affect the
predicted label, so the macro-F1 values are the same before and after scaling.
Do not refit temperatures on these development rows or generalize calibration
behavior from them.

## Paired original-order correctness

The table counts the 32 paired questions per dataset for each model pair.

| Pair (left vs right) | Dataset | Both correct | Left only correct | Right only correct | Both wrong |
|---|---|---:|---:|---:|---:|
| Qwen vs Intern | BoolQ | 15 | 3 | 11 | 3 |
| Qwen vs Intern | SNLI | 7 | 6 | 6 | 13 |
| Qwen vs Kev | BoolQ | 14 | 4 | 12 | 2 |
| Qwen vs Kev | SNLI | 10 | 3 | 12 | 7 |
| Intern vs Kev | BoolQ | 24 | 2 | 2 | 4 |
| Intern vs Kev | SNLI | 12 | 1 | 10 | 9 |

## Method, overlap and limits

The frozen [protocol](broader-baseline-protocol.md) used original-order
accuracy as primary, all semantic option permutations, and the existing model
prompts, scorers and shipped temperatures. BoolQ and SNLI each contributed 32
distinct source groups. The [runbook](broader-baseline-run.md) records source
acquisition, byte checks, preparation, the plan-only command and explicit
launch procedure.

The exact model revisions were Qwen3.5-0.8B-Base
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, Intern-Decision-0.8B
`85a0cc5a99d67ea8d56dfe98115689212867171d`, and Kev-0.8B
`9a45d25eb2ab761841196625383fa1dff0e56c1e`. The run used Torch
`2.14.1+cu130` and Transformers `5.18.0`; this Torch version is outside Kev's
upstream declared `>=2.6,<2.9` range. The pinned Kev load and parity check
passed here, which is narrow evidence for this path rather than validation of
the upstream supported range.

BoolQ is a disclosed Kev supervised source, but selected-row overlap is
unknown. Kev's MultiNLI exposure is related to SNLI, while SNLI row overlap is
unknown. Intern's training overlap and all base-pretraining overlap remain
unknown. Do not claim contamination-free or unseen-family performance. These
are development measurements, not latency, throughput, or cost benchmarks.

## Reproduction and delivery record

- Frozen code: [`cb58a35024f4019cc2e28f18790b846b670c1772`](https://github.com/1337mus/reflex/commit/cb58a35024f4019cc2e28f18790b846b670c1772); [CI run 37251982817 passed](https://github.com/1337mus/reflex/actions/runs/37251982817).
- [Canonical runner receipt](verification/broader-baselines.json), SHA-256 `dfda0723ef602d4bdce6129d4bed3c91524729073b8ad6fc5ba83b6775303de8`.
- [Canonical analysis](verification/broader-baselines-analysis.json), SHA-256 `1d81aab940d5cb6b8c8f94209e7454dd741af1e300629863eb8b0406880b4dc3`.
- [Modal teardown record](verification/broader-baseline-teardown.json), SHA-256 `a84d67b0d3f8ac1ff42b17ab62bdf091d799bb66eb724ede3b6f131c8b44a9e9`: app `ap-UeKFVEhlNZWTDuvSThqzsE` was stopped with zero tasks; it ran 18:35:48–18:37:40 PDT on 2026-10-04. App lifetime is not model latency or billed runtime, and actual cost is unverified.
- All 771 forwards passed: 256 scored presentations per model, plus one Intern and two Kev auxiliary verification forwards. No tasks remained after teardown.
- Prepared records SHA-256: `5d09ac1c1df2b7c1849e675179cb57306040f473c0ebad60527f54d186ac6845`; manifest SHA-256: `8dbddaec2ceac622ed998cd0c6051472c11bf38d776a895c08ff94ad5968b085`.

The next work is the separate bounded training-mechanics rehearsal, currently
in progress. It uses its own training fixture; BoolQ, SNLI and COPA remain
reserved for development. No Reflex adapter or training result has been
completed.
