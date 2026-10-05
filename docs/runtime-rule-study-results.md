# Routing and tool-choice study: the recipe did not pass

**Completed: 2026-10-05.** The new practice scored higher on routing, but lower on tool choice. It passed **17 of 21 preset checks**. Keep the current selected adapter. This study does not justify promoting the new one.

The training run itself passed. All planned requests completed, both saved adapters reproduced their check answers, and both the GPU app and the file-check app stopped with zero tasks.

## New skills

Routing means choosing a support queue from the supplied rules. Tool choice means selecting a tool that meets the stated needs and permissions, then choosing the lowest-cost eligible tool.

| Score | Current adapter | More existing practice | New practice |
| --- | ---: | ---: | ---: |
| Routing | 21.73% | 28.57% | 38.10% |
| Tool choice | 41.96% | 37.50% | 36.31% |
| **Average of the two skills** | **31.85%** | **33.04%** | **37.20%** |

New practice gained **4.17 percentage points** over more existing practice. We required at least 5. Its 95% uncertainty range was **−15.77 to +23.21 points**. This small panel cannot establish a reliable gain. Tool choice fell by **1.19 points** against more existing practice and **5.65 points** against the current adapter.

There are **14 question groups per new skill**. Each question appears in 4, 6, or 8 answer orders. We average each question across those orders, then average related questions within each group. Each group counts equally. The 164 new requests are not 164 independent questions. These group scores can differ from a simple count of correct requests. Displayed numbers are rounded; the checks use exact fractions.

## Where the new tasks still fail

All three versions missed **all 22 presentations of four tool-choice questions**: two had complete inputs, and two had missing inputs that still led to one tool. On two questions where no tool qualified, new practice got **7/14** presentations correct; more existing practice got **14/14**. These are tiny groups. They identify cases to study, not the cause of the failure.

Reordering the answer choices often changed the selected answer:

| Changed answer versus original order | Current adapter | More existing practice | New practice |
| --- | ---: | ---: | ---: |
| Routing | 20/68 | 19/68 | 31/68 |
| Tool choice | 21/68 | 15/68 | 38/68 |

These are 68 reorderings of 14 questions per skill. Changes can help, hurt, or move between two wrong answers. They are not accuracy scores. This check used only the existing development files. It did not read calibration/test records or change the study's rules.

## Earlier skills

| Score | Current adapter | More existing practice | New practice |
| --- | ---: | ---: | ---: |
| Topic recognition | 97.32% | 98.53% | 98.79% |
| Spam detection | 96.67% | 96.67% | 96.67% |
| Sentence reasoning | 86.46% | 89.50% | 86.63% |
| Fact rules | 98.67% | 100.00% | 100.00% |
| Number comparison | 81.25% | 90.00% | 87.88% |
| Reading comprehension | 78.12% | 84.38% | 81.25% |
| Cause and effect | 90.62% | 85.94% | 87.50% |

All **14 earlier-skill checks** passed their observed-score limit. This does not prove that every true loss is smaller than 5 points: several uncertainty ranges include larger losses. Sentence reasoning stayed at **998/1,152 correct requests (86.63%)**, versus **996/1,152 (86.46%)** for the current adapter.

## Every preset check

Values in parentheses are percentage-point differences for new practice. There are 11 rows below and 21 checks; the uncertainty check applies only to the control comparison. Four checks failed. We kept every threshold fixed.

| Rule | Versus more existing practice | Versus current adapter |
| --- | ---: | ---: |
| New-skill average: gain at least 5 points | **Fail** (+4.17) | Pass (+5.36) |
| New-skill average: uncertainty range above zero | **Fail** (−15.77 to +23.21) | Not required |
| Routing: no score drop | Pass (+9.52) | Pass (+16.37) |
| Tool choice: no score drop | **Fail** (-1.19) | **Fail** (-5.65) |
| Topic recognition: drop no more than 5 points | Pass (+0.26) | Pass (+1.47) |
| Spam detection: drop no more than 5 points | Pass (+0.00) | Pass (+0.00) |
| Sentence reasoning: drop no more than 5 points | Pass (-2.86) | Pass (+0.17) |
| Fact rules: drop no more than 5 points | Pass (+0.00) | Pass (+1.33) |
| Number comparison: drop no more than 5 points | Pass (-2.12) | Pass (+6.62) |
| Reading comprehension: drop no more than 5 points | Pass (-3.12) | Pass (+3.12) |
| Cause and effect: drop no more than 5 points | Pass (+1.56) | Pass (-3.12) |

All 20 uncertainty ranges are below. Each cell shows the observed difference, then its 95% range. We used the fixed 2,000 paired resamples of source groups. These ranges describe this panel and this one training seed.

| New practice minus comparison, points | More existing practice | Current adapter |
| --- | ---: | ---: |
| New-skill average | +4.17 [-15.77, +23.21] | +5.36 [-11.76, +22.03] |
| Routing | +9.52 [-22.92, +41.67] | +16.37 [-10.72, +42.57] |
| Tool choice | -1.19 [-20.24, +21.73] | -5.65 [-23.51, +16.67] |
| Topic recognition | +0.26 [-0.45, +0.96] | +1.47 [+0.00, +4.15] |
| Spam detection | +0.00 [+0.00, +0.00] | +0.00 [-2.50, +2.50] |
| Sentence reasoning | -2.86 [-6.34, +0.52] | +0.17 [-2.52, +2.69] |
| Fact rules | +0.00 [+0.00, +0.00] | +1.33 [+0.44, +2.22] |
| Number comparison | -2.12 [-6.25, +1.25] | +6.62 [+1.38, +12.25] |
| Reading comprehension | -3.12 [-14.06, +4.73] | +3.12 [-6.25, +12.50] |
| Cause and effect | +1.56 [+0.00, +4.69] | -3.12 [-9.38, +1.56] |

## What changed, and what was verified

Both runs started from the same selected Qwen3.5-0.8B adapter. Both used 336 learning steps and 1,344 training presentations. They shared 1,008 presentations. New practice replaced the other 336 with 56 routing and 56 tool-choice examples, each shown three times. The control repeated existing practice. Both used the same fresh optimizer and learning rate. Only step 336 was scored; step 168 was saved without scoring.

| Check | Result |
| --- | ---: |
| Planned model passes completed | 10,808 / 10,808 |
| Total input tokens | 2,544,210 |
| Training tokens: control / new practice | 225,924 / 349,296 |
| Base-model weights | Stayed frozen |
| Adapter tensors changed, per run | 120 / 120 |
| Answers reproduced after reload, per run | 32 / 32 |
| Largest score difference after reload | 0 |
| Saved files independently hashed on Modal | 14 / 14 |
| New calibration or test files used | 0 |

New practice used about **55% more training text**. This comparison measures the whole replacement recipe. It does not isolate text volume, example variety, or practice quality. The laptop did not load model weights or run the model. The file check returned hashes and sizes only.

The run used the signed [source commit 6fa151e](https://github.com/1337mus/reflex/commit/6fa151e4d10ed61d44da35a89e89d8dbe751c5be). Its [GitHub checks passed all 1,178 tests](https://github.com/1337mus/reflex/actions/runs/37314334195). A separate calculation matched every task score, all 21 checks, and all 20 uncertainty ranges. The [verification summary](verification/runtime-rule-study-summary.json) records the exact hashes, saved paths, counts, and results.

Recorded workspace usage after the study was **$2.91 before credits; $0 billed after credits**. This is a delayed workspace total, not the cost of this run alone. The pre-launch worker estimate was $3.32 with a $10 reserve under the roughly $300 project allocation. Build, storage, network, and tax charges are outside that estimate.

## Decision and limits

Keep the selected adapter from the earlier reasoning study. Preserve both new adapters and this failed result. Do not retry this recipe, change the checks, pick the halfway save, or claim broad improvement from this run.

The new panels are small and were already inspected during development. Each has only two cases with several missing inputs; those cases agree across all possible completions. Cases where those completions disagree are missing from this panel. Calibration and test sets remain sealed. This is one training seed, with no measured seed uncertainty.

The development-only error check points to a useful next step: separate basic tool eligibility and cost selection from missing-information cases, then define a new data plan and fixed comparison. This study gives no reason to spend more GPU time repeating the same recipe. Its result does not change any earlier failed result.
