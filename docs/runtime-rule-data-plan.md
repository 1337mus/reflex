# Next practice data: routing and tool choice

**Status (2026-10-05): support-routing v1 is generated and independently checked; the tool-choice foundation is implemented and independently reviewed.** Its signed, pushed [source commit](https://github.com/1337mus/reflex/commit/6b0dc56b5cc84f6e79357b91f2e5570f597b207c) passed review with no findings and exact-commit [CI](https://github.com/1337mus/reflex/actions/runs/37290214455). Tool-choice data implementation is underway, but no tool-choice release dataset has been generated. No model result exists yet. See the [foundation report](tool-choice-foundation-results.md) and [verification summary](verification/tool-choice-foundation-summary.json).

## Proposed answer rules

Each scenario format has one fixed menu. Include every menu option in every record, whether it is correct or a distractor. Use 2–6 destination or tool names plus two fixed special answers. This keeps the menu between 4 and 8 options. Shuffle choices from scenario identity, independently of the answer.

| Family | Program-derived answer |
| --- | --- |
| Support routing | Match a complete attribute key to one route. For a missing attribute, solve every declared completion. Return the common route or “No route applies” when all completions agree; return “Insufficient information” when outcomes differ. |
| Tool choice | Keep tools that meet every required capability, input type, and permission. Choose the unique lowest-cost tool. Return “No eligible tool” if none qualify; for missing permissions, return “Insufficient information” when possible outcomes differ. Reject ties. |

Keep correct answers and calculation details outside model requests. A separate parser must read the rules and values back from each prompt and compare them with the original structured scenario. Checking the stored values alone does not prove that the prompt kept the same meaning. Set aside text that cannot pass this check for independent review.

## Proposed starting sample

Start with one decision per scenario. Counts apply once to each family, for 224 records when both families are present.

| Per family | Train | Development | Calibration | Test | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scenarios and records | 56 | 14 | 14 | 28 | 112 |

Split every source scenario and its related questions as one group. Hash the canonical rules, candidates, constraints, and query. Also hash each underlying rule or capability set so sibling queries cannot cross splits. Exclude wording, split, option order, and generated IDs from these hashes. Reject duplicates across splits. Reserve named test-only templates and domain values, then audit they do not occur in training.

The support-routing release uses four binary attributes (two allowed values each), which provides enough structural variety for its current split schedule. Its independent release check found 112 rows in 112 groups, no cross-split equivalent rule tables, all 112 answers matching an independent recount, and all 11 source hashes matching. The training split covers eight of nine menu-size and two-missing-outcome combinations, with multiple outcome classes at each menu size. See [the release report](routing-data-results.md) and [the verification summary](verification/routing-data-summary.json) for exact pins and hashes.

The 28 test cases per family can expose basic failures. They are too few for a reliable estimate of broad performance. Keep test requests and answers sealed during development. Use training and development examples to debug generation. The support-routing release has a sealed test split; tool-choice data generation and its independent audit remain to be completed.

## Implementation and later evaluation

Tool-choice data implementation is underway. Verify prompts, answers, grouping, split independence, and release metadata when the data is generated, before combining the families or beginning a model study. The synthetic answer program does not run real tools, and its checks do not establish model performance.

A no-update adapter is only a mechanical save, reload, and scoring check. To study the data effect later, use equal presentations and updates: replace repeated existing practice with new-family examples in the treatment, while the continued-training control keeps existing practice. Include retention results. The proposed counts are not a run plan, and success on these synthetic cases would not establish broad reasoning ability.
