# Next practice data: routing and tool choice

**Status: the CPU-only deterministic support-routing answer program is implemented and checked.** Tool-choice support, new-family generators, the prompt parser, data generation, and training remain pending. No teacher model was used; the proposed counts remain provisional.

The current generator covers atomic fact labels and numeric minimum/maximum choices. It has no populated test split. The request schema accepts 2–16 options; this proposal uses fixed menus of 4–8 options. It tests whether a model can follow rules and capabilities stated in each request. It does not test arbitrary typed schemas or prove transfer to real work.

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

The 28 test cases per family can expose basic failures. They are too few for a reliable estimate of broad performance. Keep test requests and answers sealed during development. Use training and development examples to debug generation. The current data format supports a test split, but the synthetic generator does not create one today.

## Implementation and later evaluation

Implement support routing first, in a code slice of at most 500 production lines. Verify its generator, parser, solver, and split audits before adding tool choice. Tool choice follows as a separate slice.

A no-update adapter is only a mechanical save, reload, and scoring check. To study the data effect later, use equal presentations and updates: replace repeated existing practice with new-family examples in the treatment, while the continued-training control keeps existing practice. Include retention results. The proposed counts are not a run plan, and success on these synthetic cases would not establish broad reasoning ability.
