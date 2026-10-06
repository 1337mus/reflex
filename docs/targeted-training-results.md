# Targeted HANS and WinoGrande training results

The new practice improved both reserved tasks in this study, but it also
lowered accuracy on several earlier tasks. It failed the fixed acceptance
checks: 19 of 24 passed. Keep the previously selected `snli_mix` adapter at
update 378. Keep the new adapter as an experiment.

## Reserved tasks

All three states used the same reserved questions: the earlier adapter, more
earlier practice, and the new practice.

| Task | Questions | Earlier adapter | More earlier practice | New practice |
| --- | ---: | ---: | ---: | ---: |
| Sentence logic (HANS) | 300 | 63.3% | 71.7% | 88.3% |
| Commonsense (WinoGrande) | 200 | 54.0% | 51.0% | 63.3% |

Each question was tested in two answer orders. We averaged the two orders
within each pre-set group, then gave every group equal weight. A point means
one percentage point.

| Task | New practice vs more earlier practice | 95% uncertainty range |
| --- | ---: | ---: |
| Sentence logic (HANS) | +16.7 points | +5.2 to +28.7 |
| Commonsense (WinoGrande) | +12.3 points | +5.8 to +19.0 |

The ranges compare paired groups and show sampling uncertainty. These are the
R2 reserved questions, not the earlier fresh-evaluation questions or base-Qwen
scores. The gains are clear on these samples, but do not prove broad results.

## Sentence logic claim types (HANS)

Each HANS class below has 150 questions, shown twice per state in the two
answer orders. The new practice lost ground on supported claims and improved
on unsupported claims.

| HANS claim type | Earlier adapter | New practice | Change |
| --- | ---: | ---: | ---: |
| Supported claims | 94.7% | 80.3% | −14.3 points; check failed |
| Unsupported claims | 32.0% | 96.3% | +64.3 points |

## Earlier skills

The fixed limit allowed at most a 5-point loss against both other states.
These are simple checks against the study rule, not proof of lasting damage.

| Earlier task | Earlier adapter | More earlier practice | New practice |
| --- | ---: | ---: | ---: |
| Science (ARC) | 67.8% | 66.5% | 62.0% |
| Reading (BoolQ) | 78.1% | 76.6% | 71.9% |
| Cause and effect (COPA) | 90.6% | 85.9% | 71.9% |

Science and reading passed against more earlier practice but failed against
the earlier adapter. Cause and effect failed against both. These are five
failed checks in total: supported HANS claims, science, reading, and both
cause-and-effect comparisons. The other five earlier tasks stayed within the
fixed loss limits; their values and all 24 checks are in the
[verification record](verification/targeted-training-summary.json).

Cause and effect had the largest loss. These results do not show which part
of the new practice caused it.

## Training and limits

Both training runs used 1,200 updates. The new practice replaced one quarter
of earlier practice with 600 HANS and 600 WinoGrande examples. This was one
training seed, with shared HANS templates. It cannot separate the effects of
the two datasets. Exact-overlap checks do not rule out paraphrases or prior
exposure during pretraining. There is no overall reasoning score.

## Recovery and evidence

The original controller stopped with `KeyboardInterrupt` before it wrote a
success receipt. The run remains recorded as failed. All three saved worker
results passed rechecking and were admitted under a separately reviewed
recovery amendment.

The V1 independent recount had a Unicode request-hash bug. We found it after
the first official result calculation, before the independent checker calculated scores. V2 changed the checker only. It produced exactly
the same metrics and checks as V1, and the corrected independent recount
agreed exactly. No study data or rules changed.

We do not know why the controller stopped or whether the provider made extra
attempts or model passes. Save and reload checks were recorded during the run.
Later storage checks verified file paths and sizes, not a fresh remote weight
hash. Recovery made no new model calls. The Modal app was stopped with zero
tasks. The latest delayed workspace snapshot showed $4.53 metered and $0
billed; it is workspace usage, not a per-run cost.

The signed analysis source was `d936caa97db07e129aabd8e421bce556b3de114f`.
Exact-source CPU CI run 37409478235 passed. The V2 amendment is revision 2.
See the [verification record](verification/targeted-training-summary.json)
for exact pins, counts, checks, and evidence.

Any new practice recipe is untested. These reserved questions are now open,
so evaluate a changed recipe on fresh held-out questions. No new run is
proposed or launched here.
