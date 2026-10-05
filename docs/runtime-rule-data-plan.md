# Next practice data: routing and tool choice

**Status (2026-10-05): support-routing v1 and tool-choice v1 are generated and independently checked.** The tool-choice foundation's signed [source commit](https://github.com/1337mus/reflex/commit/6b0dc56b5cc84f6e79357b91f2e5570f597b207c) passed review with no findings and exact-commit [CI](https://github.com/1337mus/reflex/actions/runs/37290214455). The tool-data source then passed review and exact-commit CI before one fresh release seed was selected; the seed was not searched. Both 112-record datasets are ready for a fixed matched continued-training study. No model has been scored or trained on them. See the [tool-choice release report](tool-data-results.md) and [verification summary](verification/tool-data-summary.json), plus the [tool-choice foundation report](tool-choice-foundation-results.md) and [foundation verification summary](verification/tool-choice-foundation-summary.json).

## Proposed answer rules

Each scenario format has one fixed menu. Include every menu option in every record, whether it is correct or a distractor. Use 2–6 destination or tool names plus two fixed special answers. This keeps the menu between 4 and 8 options. Shuffle choices from scenario identity, independently of the answer.

| Family | Program-derived answer |
| --- | --- |
| Support routing | Match a complete attribute key to one route. For a missing attribute, solve every declared completion. Return the common route or “No route applies” when all completions agree; return “Insufficient information” when outcomes differ. |
| Tool choice | Keep tools that meet every required capability, input type, and permission. Choose the unique lowest-cost tool. Return “No eligible tool” if none qualify; for missing permissions, return “Insufficient information” when possible outcomes differ. Reject ties. |

Keep correct answers and calculation details outside model requests. A separate parser must read the rules and values back from each prompt and compare them with the original structured scenario. Checking the stored values alone does not prove that the prompt kept the same meaning. Set aside text that cannot pass this check for independent review.

## Verified starting sample

Each family has one decision per scenario. The counts apply once to each family, for 224 records across both verified datasets.

| Per family | Train | Development | Calibration | Test | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scenarios and records | 56 | 14 | 14 | 28 | 112 |

Split every source scenario and its related questions as one group. Hash the canonical rules, candidates, constraints, and query. Also hash each underlying rule or capability set so sibling queries cannot cross splits. Exclude wording, split, option order, and generated IDs from these hashes. Reject duplicates across splits. Reserve named test-only templates and domain values, then audit they do not occur in training.

The support-routing release uses four binary attributes (two allowed values each), which provides enough structural variety for its current split schedule. Its independent release check found 112 rows in 112 groups, no cross-split equivalent rule tables, all 112 answers matching an independent recount, and all 11 source hashes matching. The training split covers eight of nine menu-size and two-missing-outcome combinations, with multiple outcome classes at each menu size. See [the release report](routing-data-results.md) and [the verification summary](verification/routing-data-summary.json) for exact pins and hashes. Tool-choice v1 has 112 rows in 112 distinct catalog groups, all seven case types, and 112 independently recomputed answers; its recipe pins 12 source files. Its 28 sealed test cases remain a small synthetic evaluation set, not evidence of broad performance. See [its release report](tool-data-results.md) and [verification summary](verification/tool-data-summary.json).

The 28 test cases per family can expose basic failures. They are too few for a reliable estimate of broad performance. Keep test requests and answers sealed during development. Use training and development examples to debug generation. Both releases have sealed test splits; release generation and independent audits are complete.

## Implementation and later evaluation

Both datasets have passed their release checks. The next step is a fixed matched continued-training study, with equal presentations and optimizer updates between treatment and continued-practice control, plus retention checks. Keep test requests and answers sealed until the evaluation is fixed. The synthetic answer programs do not run real tools, and release checks do not establish model performance.

Keep an unchanged adapter as a reference and to check saved weights and scoring. Use the continued-training control to measure the new practice effect. Treatment replaces repeated existing practice with new-family examples; match presentations and optimizer updates, then measure retention. Release counts alone do not specify the run plan or establish broad reasoning ability.
