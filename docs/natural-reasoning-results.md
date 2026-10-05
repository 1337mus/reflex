# Human-labeled reasoning practice: results

**The new adapter scored 86.46%, compared with 66.75% for the matched control.**
All nine checks set before training passed. We will use this saved adapter for
the next cause-and-effect and reading-comprehension checks.

These are development results from SNLI, a dataset now used in both training
and evaluation. They show an improvement on this source. They do not establish
general reasoning ability or performance on a sealed test.

## What changed

Both versions started from identical saved weights. Each received 1,512
training presentations and 378 weight updates. The model, learning rate,
shared examples, and evaluation questions stayed the same.

| Practice given to each version | Control | New adapter |
| --- | ---: | ---: |
| Real topic and spam examples | 504 | 504 |
| Generated logic examples | 1,008 | 504 |
| Human-labeled SNLI examples | 0 | 504 |

The new adapter replaced a repeated pass over generated examples with 500
human-labeled reasoning examples plus four repeats. Related source groups
that overlapped the complete SNLI development split were excluded from
training. The official test split remained unopened.

## Reasoning accuracy

The panel contains **192 questions**, each shown in **six answer orders**.
That gives 1,152 presentations, from 64 groups of related questions.

| Version | Correct / 1,152 | Accuracy |
| --- | ---: | ---: |
| Qwen base | 447 | 38.80% |
| Control: more generated practice | 769 | 66.75% |
| New adapter: human-labeled practice | **996** | **86.46%** |

The gain over the control was **19.70 percentage points**. Its paired 95%
interval was **+14.32 to +24.48 points**. This interval resamples the 64 groups;
it does not measure variation between different training runs.

The original answer order shows where performance changed:

| Type of claim | Control | New adapter |
| --- | ---: | ---: |
| Conflicts with the given facts | 23/64 | **58/64** |
| Follows from the given facts | **60/64** | 57/64 |
| Not enough information to decide | 37/64 | **52/64** |

The earlier second-seed synthetic adapter got 15/64 unknown cases correct.
That older run had fewer updates and a different practice schedule. Use the
37/64 matched control above to assess this data change.

For context, earlier measurements on this same panel were **58.16% for Intern**
and **76.13% for Kev**. Their prompts and runtime precision differ. Our new
86.46% score is higher on this development panel; it is not evidence of broad
superiority. Base pretraining overlap is unknown.

## Skills retained and tradeoffs

In the original answer order, the new adapter got **55/56 topic questions**
and **58/60 spam questions** correct. The control got 53/56 and 58/60.

Across the full set of answer orders, the new adapter had these small losses
against the control:

| Task | Change in accuracy | Allowed loss | Result |
| --- | ---: | ---: | --- |
| Topic classification | −0.57 points | 5 points | Pass |
| Spam detection | −0.83 points | 5 points | Pass |
| Generated fact questions | −1.33 points | 5 points | Pass |
| Generated number questions | −3.50 points | 5 points | Pass |

The number-task score gives each source group equal weight. It first averages
answer orders within each question. This keeps questions with more answer
choices from dominating the score.

The five other checks also passed: reasoning gain over the control, reasoning
retention against the base, topic and spam retention against the saved earlier
adapter, and retention on the original SNLI panel. The new adapter scored
120/128 on that original panel, versus the fixed base reference of 57/128.

Retention checks use the observed difference, as set before the run. They do
not prove that losses are below five points on a larger population. For
example, the number-task interval was −8.25 to +0.75 points. The earlier
real-data pilot's failed reasoning check remains a failure.

## Verification and next step

- Both runs completed all 378 updates. All 120 adapter tensors changed; the base stayed frozen.
- Fresh reloads reproduced exact adapter tensors, all 32 checked answers, and identical candidate scores in each run.
- A separate calculation matched 36 score counts, 36 group means, six confusion matrices, all nine checks, and their intervals.
- All 15 saved files passed remote hash checks. Both Modal apps stopped with zero tasks.
- The run used 12,073 model forwards and 2,594,387 input tokens. Equal presentation counts did not mean equal training tokens: 308,495 for the control and 257,408 for the new adapter.

The account snapshot showed **$2.31 metered and $0 billed after credits**.
This is delayed, revisable workspace billing, not the price of this run.
All model work ran on personal Modal. No weights were downloaded to the laptop.

Next, compare the selected adapter with a fresh base on the existing COPA and
BoolQ development panels. These questions were excluded from adapter training,
but we have inspected the panels before. They are not sealed tests.

See the [frozen training protocol](natural-reasoning-protocol.md),
[aggregate verification evidence](verification/natural-reasoning-summary.json),
and [next evaluation plan](adapter-transfer-protocol.md).
Reproduction requires source commit
`3143e4a0533b3ed111fd1fcba398e652a41f4b46`; its CPU CI passed before launch.
Private receipts use the prefix `artifacts/natural-reasoning-2026-10-04-r1-`.
