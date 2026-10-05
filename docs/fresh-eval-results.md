# 700 new public benchmark questions

**The clearest gain is in science questions. Commonsense remains weak.**
The saved adapter also scored higher on sentence logic, but that gain has a wide
uncertainty range. These results show why the earlier
[86.5% SNLI score](natural-reasoning-results.md) should not be treated as an
overall reasoning score.

We compared the unchanged Qwen3.5-0.8B base with our selected adapter. We performed
no training. Each question appeared in its original answer order and once with
the answers moved by one position. The table averages those two scores.

| Task | Questions | Base correct | Our adapter correct |
| --- | ---: | ---: | ---: |
| Sentence logic — HANS | 300 | 304/600 (50.7%) | **385/600 (64.2%)** |
| Commonsense — WinoGrande | 200 | 197/400 (49.3%) | **206/400 (51.5%)** |
| Science — ARC-Challenge | 200 | 209/400 (52.3%) | **274/400 (68.5%)** |

The doubled denominators count two answer orders. They do not mean twice as many
independent questions. WinoGrande has two choices, so 51.5% is near chance.
ARC has four choices in 198 questions and three choices in two questions.

## How clear are the changes?

A percentage point is the difference between two percentages. The 95% ranges
below estimate uncertainty in the adapter's change from the base. A range that
includes zero does not show a clear gain on this sample under this method.

| Task | Change across both orders | 95% range |
| --- | ---: | ---: |
| Sentence logic | +13.5 points | −0.2 to +26.7 |
| Commonsense | +2.3 points | −3.0 to +7.3 |
| Science | **+16.3 points** | **+10.8 to +21.8** |

HANS uses 30 sentence patterns. Its range resamples these patterns as groups,
because ten questions share each pattern. The other ranges resample questions.
Each draw keeps both models and both answer orders together. We used the fixed
2,000-draw method from the [protocol](fresh-eval-protocol.md).

## Original answer order

| Task | Base correct | Our adapter correct | Change | 95% range |
| --- | ---: | ---: | ---: | ---: |
| Sentence logic | 154/300 (51.3%) | 192/300 (64.0%) | +12.7 points | −17.0 to +42.0 |
| Commonsense | 97/200 (48.5%) | 99/200 (49.5%) | +1.0 point | −7.0 to +8.5 |
| Science | 103/200 (51.5%) | 137/200 (68.5%) | +17.0 points | +10.0 to +24.0 |

## Does moving the choices change the answer?

This counts questions where the model chose a different *answer*, not just a
different displayed letter. Lower counts mean less dependence on answer position.

| Task | Base changed its answer | Our adapter changed its answer |
| --- | ---: | ---: |
| Sentence logic | 294/300 (98.0%) | **7/300 (2.3%)** |
| Commonsense | 155/200 (77.5%) | **70/200 (35.0%)** |
| Science | 93/200 (46.5%) | **41/200 (20.5%)** |

More stable answers can still be wrong. WinoGrande shows that clearly: answer
changes fell, but accuracy stayed near chance.

## Where sentence logic breaks

HANS checks three tempting shortcuts. Its questions test when each shortcut works
and when it gives the wrong answer. These are descriptive breakdowns.

| Shortcut under test | Base, both orders | Our adapter, both orders |
| --- | ---: | ---: |
| Shared words imply shared meaning | 103/200 (51.5%) | 149/200 (74.5%) |
| A shorter word sequence must follow from the sentence | 101/200 (50.5%) | 110/200 (55.0%) |
| A phrase inside a sentence must be true on its own | 100/200 (50.0%) | 126/200 (63.0%) |

A further breakdown after the run found a strong imbalance. When a claim did
follow from the sentence, the adapter scored **285/300 (95.0%)** across both
orders. When it did not follow, the adapter scored **100/300 (33.3%)**. Each side
contains 150 questions shown twice. This suggests a tendency to accept claims
that the sentence does not support. It does not establish the cause of that bias.

All 30 sentence-pattern breakdowns, including original scores and answer changes,
are retained in the [verification summary](verification/fresh-eval-summary.json).
Each pattern has only ten questions; avoid ranking patterns from such small counts.

## What this run establishes

The code completed **2,824 scoring passes** and zero training steps on one Modal
A10. Reloading the saved adapter reproduced all 24 checked answers with exactly
equal saved tensors and zero score difference. An independent recount checked
the task counts, uncertainty ranges, and all HANS breakdowns.

The app stopped with zero tasks. The saved workspace billing snapshot showed
**$2.84 metered and $0 billed after credits**. Billing can lag or change. This is
total workspace usage, not the price of this run. The prelaunch reserve was $3.

These public benchmark questions were newly added to this project. Exact-match
checks found no overlap with the selected adapter's 1,504 training records.
That does not rule out paraphrases or exposure during Qwen's earlier training.
One known ARC preview was excluded. Other preview overlap remains unknown.
The project's separate calibration and test pools stayed closed.

This measures the whole selected adapter against its base. It does not isolate
which training change caused the gains. We did not run Intern or Kev on these
questions, so this report does not rank our adapter against them. The earlier
failed training studies remain failed. No new pass threshold or overall
reasoning score was introduced.

Sources: [HANS](https://github.com/tommccoy1/hans),
[WinoGrande](https://github.com/allenai/winogrande), and
[ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc).
See the source and license notes for [HANS](licenses/fresh-eval-hans.md),
[WinoGrande](licenses/fresh-eval-winogrande.md), and
[ARC](licenses/fresh-eval-arc.md). The private report contains aggregate results;
raw questions and individual predictions remain local artifacts.
