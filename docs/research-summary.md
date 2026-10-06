# Reflex research summary

**Updated:** 2026-10-06

Reflex studies whether Qwen3.5-0.8B-Base can choose among two to sixteen options supplied with a request, using one model
pass and no generated rationale or answer text.

**Status:** research prototype. The selected adapter remains `snli_mix` at update 378.

Three findings define the current picture:

- In a 700-question public evaluation, science had a clear measured gain; the sentence logic and commonsense ranges
included zero. See the [fresh evaluation](fresh-eval-results.md).
- A human-labeled SNLI training comparison scored 86.46% versus 66.75% on SNLI development data from that same
source. It is not a held-out reasoning result. See [those results](natural-reasoning-results.md).
- New HANS/WinoGrande practice improved reserved questions but failed retention checks: 19 of 24 passed. Keep
`snli_mix` at update 378. See the [targeted study](targeted-training-results.md).

Most panels below are small development samples; targeted HANS and WinoGrande questions were reserved for that study. These
results do not establish general transfer, calibrated correctness, a speed or cost advantage, or release readiness.

## How to read the results

Each study uses its own questions, options, answer orders, source groups, and scoring rules. Accuracy values from different
tables should not be pooled. A second answer order is a paired view of the same question, not a new independent example.

Some later studies average each question over its answer orders, then average within source groups and give each group equal
weight. Their group means can differ from raw correct/presentation counts. The tables show group scores where stated;
the linked verification files retain exact fractions and raw counts.

The reported confidence intervals estimate sampling variation over questions or source groups, as stated for each study. They
do not measure variation between training seeds. An interval that includes zero does not establish a gain on that sample
under the reported method.

Normalization makes candidate scores sum to one over the options in a request. It does not by itself make them probabilities
that the selected answer is correct. Some studies fit a temperature on a designated calibration split; those are calibration
fits, not proof of calibration on new data.

Early pilot fits used 116 DBpedia/SMS calibration rows, and the controlled-mixture fits used 241 rows. The runtime-rule study
used no new calibration or test files; its separate calibration and test pools remained closed.

## Terms

- **LoRA:** Low-Rank Adaptation, a small set of trainable adapter weights used with a frozen base model.
- **Logit:** a raw numeric score for a candidate option before normalization.
- **Percentage point:** the subtraction of two percentages. A change from 50% to 60% is a gain of 10 percentage
points.
- **Answer-order view:** the same question with its options shown in a different order. A semantic answer flip means
the selected option changed, even if its displayed letter also changed.
- **NLL:** negative log likelihood. It penalizes a low probability for the correct answer. Lower is better.
- **Brier score:** for each question, sum squared probability errors across all choices, then average those sums over
questions. Lower is better for the scored sample.
- **ECE:** expected calibration error, here a ten-bin comparison of confidence and accuracy. It is noisy on small
panels and depends on the binning rule.

See the [CPU foundation](cpu-foundation.md) for the evaluator definitions, including coverage, accepted risk, macro
averaging, and calibration limits.

## 1. Prior-art baselines

The first frozen comparisons used public Qwen3.5-0.8B-Base, Intern-Decision-0.8B, and Kev-0.8B on small development panels.
No Reflex adapter had been trained yet. These are results for the named systems, not Reflex results or a broad model ranking.

### COPA

The 32-question cause-and-effect panel was scored in its original and reversed answer orders. The table gives original-order
correct answers and semantic answer flips across the paired orders.

| Model | Original order | Reversed order | Changed answer / 32 |
| --- | ---: | ---: | ---: |
| Qwen3.5-0.8B-Base | 22/32 (68.8%) | 13/32 | 29 |
| Intern-Decision-0.8B | 26/32 (81.3%) | 30/32 | 8 |
| Kev-0.8B | 31/32 (96.9%) | 31/32 | 0 |

The COPA answer key contains 11 `choice1` and 21 `choice2` labels; always choosing `choice2` would score 21/32. Qwen chose
the second presented option on 31/32 original and 30/32 reversed questions. Kev's zero flips apply only to these 32 pairs.

The shipped temperatures for Intern and Kev were 2.747760550703 and 2.3510958125672174. On this 32-question panel, their
shipped temperatures worsened NLL, Brier score, and ECE compared with raw scores. ECE is especially noisy at this size. The
detailed metrics and intervals are in the [COPA report](copa-baseline-results.md) and [verification
analysis](verification/copa-baselines-analysis.json).

### BoolQ and SNLI

The broader pilot used 32 question groups per dataset. BoolQ used two answer orders; SNLI used all six orders for its three
labels. The counts below use original order for accuracy. “Groups with a flip” counts groups where at least one measured
order changed the selected semantic answer.

| Model | Dataset | Original order | Groups with a flip |
| --- | --- | ---: | ---: |
| Qwen3.5-0.8B-Base | BoolQ | 18/32 (56.3%) | 18/32 |
| Intern-Decision-0.8B | BoolQ | 26/32 (81.3%) | 4/32 |
| Kev-0.8B | BoolQ | 26/32 (81.3%) | 2/32 |
| Qwen3.5-0.8B-Base | SNLI | 13/32 (40.6%) | 32/32 |
| Intern-Decision-0.8B | SNLI | 13/32 (40.6%) | 24/32 |
| Kev-0.8B | SNLI | 22/32 (68.8%) | 2/32 |

BoolQ is a disclosed Kev supervised source with unknown selected-row overlap. Kev has related MultiNLI exposure to SNLI, and
overlap is unknown. Other pretraining overlap was not established. The full [BoolQ/SNLI report](broader-baseline-results.md)
links the exact source groups, intervals, macro-F1, and probability metrics.

## 2. LoRA mechanics rehearsal

A bounded rehearsal used a separate self-authored fixture of 64 records across two tasks. Every record was used for training,
then evaluated in original and reversed order. The exercise tested training, saving, and reloading; it was not a benchmark.

| Slice | Qwen base | Update 16 | Fresh reload |
| --- | ---: | ---: | ---: |
| All presentations / 128 | 101/128 (78.9%) | 128/128 (100%) | 128/128 (100%) |
| Customer support / 64 | 42/64 | 64/64 | 64/64 |
| Infrastructure / 64 | 59/64 | 64/64 | 64/64 |
| Original answer order / 64 | 54/64 | 64/64 | 64/64 |
| Reversed answer order / 64 | 47/64 | 64/64 | 64/64 |

The run completed 16 updates. LoRA trained 2,015,232 parameters across 60 modules; all 120 adapter tensors changed while the
BF16 base stayed frozen. A fresh reload reproduced all 128 selected answers, exact adapter tensors, and a maximum logit
difference of zero.

The training and evaluation records were the same fixture, so this shows memorization and training mechanics only. See the
[rehearsal report](training-rehearsal-results.md) and its [verification receipt](verification/training-rehearsal.json).

## 3. Real-data LoRA pilot: failed SNLI gate

The R2 pilot trained on DBpedia-14 and SMS Spam and checked non-regression on SNLI. Original answer order is primary;
denominators are source groups. Flip counts show groups whose selected semantic answer changed in any measured order.

| Task | Groups | Base | Final adapter | Base flips | Adapter flips |
| --- | ---: | ---: | ---: | ---: | ---: |
| DBpedia-14 | 56 | 31/56 (55.4%) | 50/56 (89.3%) | 56/56 | 11/56 |
| SMS Spam | 60 | 26/60 (43.3%) | 58/60 (96.7%) | 49/60 | 0/60 |
| SNLI | 128 | 57/128 (44.5%) | 1/128 (0.8%) | 128/128 | 99/128 |

Equal-task DBpedia/SMS macro accuracy rose from 49.35% to 92.98%, a gain of 43.63 points (paired source-group 95% interval
+33.51 to +53.15). SNLI accuracy fell by 43.75 points (95% interval −52.34 to −34.38), beyond the preregistered maximum loss
of five points. The overall learning gate failed despite the DBpedia/SMS gains.

Temperatures were fit on 116 original-order DBpedia/SMS calibration rows using an 82-candidate grid. These were in-sample
calibration diagnostics, not held-out estimates.

| State | Temperature | Calibration NLL, raw | Calibration NLL, fit |
| --- | ---: | ---: | ---: |
| Base | 0.921587 | 1.253614 | 1.251930 |
| Final adapter | 1.672640 | 0.503486 | 0.375419 |

An audit found that all 128 selected SNLI representatives had the lowest `pairID` among rows from the same caption, and 124
caption sets mixed labels. This is a serious selection warning, not proof of the cause of the loss. R1 is excluded because
training never started and no result was persisted. See the [pilot report](real-pilot-results.md) and [source
notices](licenses/real-pilot-sources.md).

### Reference models on the same pilot rows

Intern and Kev were also scored on the selected DBpedia, SMS, and SNLI rows. The table's order is original-order correct /
groups; it does not rank overall systems.

| Model | DBpedia-14 | SMS Spam | SNLI |
| --- | ---: | ---: | ---: |
| Qwen final adapter | 50/56 | 58/60 | 1/128 |
| Intern Decision | 51/56 | 52/60 | 8/128 |
| Kev | 56/56 | 45/60 | 58/128 |

For DBpedia/SMS macro accuracy, both paired intervals for reference-minus-Qwen included zero: Intern −4.11 points (−12.03 to
+3.69), Kev −5.48 points (−13.04 to +1.97). No overall winner was assigned. Kev disclosed DBpedia and related MultiNLI
exposure; row overlap is unknown. See the [matched comparison](real-pilot-baselines-results.md).

## 4. Balanced-SNLI diagnostic

This separate development panel contains 192 questions from 64 source groups, with three related questions per group. Each
question was scored in all six answer orders, for 1,152 presentations per state. The interval resamples source groups, not
orders or training seeds.

| Model state | Correct / 1,152 (95% group interval) | Original order / 192 | Changed answer / 192 |
| --- | ---: | ---: | ---: |
| Qwen base | 447 (38.80%; 37.41–40.19%) | 96 (50.0%) | 192 |
| Qwen R2 adapter | 418 (36.28%; 34.90–37.76%) | 64 (33.3%) | 65 |
| Intern-Decision-0.8B | 670 (58.16%; 54.86–61.63%) | 96 (50.0%) | 139 |
| Kev-0.8B | 877 (76.13%; 70.83–81.25%) | 143 (74.5%) | 20 |

The adapter-minus-base difference was −2.52 points (95% group interval −4.25 to −0.87). On original-order inputs, the adapter
predicted entailment for all 192 records, getting 64 correct. Fewer changes did not mean better decisions here. The earlier
real-data pilot's failed SNLI gate remains failed. This is a development comparison; prompt, tokenizer, and precision also
differ between systems. See the [diagnostic report](snli-diagnostic-results.md).

## 5. Controlled-mixture training: two seeds

The control and mixture arms were compared on the same 192-question balanced-SNLI panel in all six answer orders. Each row is
a separate initialization. Its interval resamples 64 source groups and does not measure variation between training seeds.

| Run | Real-only control | Synthetic mixture | Gain over control, 95% interval |
| --- | ---: | ---: | ---: |
| R1 | 617/1,152 (53.56%) | 781/1,152 (67.80%) | +14.24 points (+10.76 to +17.71) |
| R2 | 493/1,152 (42.80%) | 748/1,152 (64.93%) | +22.14 points (+18.49 to +25.26) |

Each arm received 1,008 training presentations over 252 updates. The mixture replaced 504 real examples with 500 checked
synthetic examples plus four repeats. Both runs passed their seven predeclared checks. Keep the seed results separate; two
seeds do not establish seed uncertainty or broad generalization. Kev scored 76.13% on this same panel in the earlier
diagnostic, ahead of both mixture runs.

The controlled runs fit temperatures on 241 original-order calibration rows: 56 DBpedia, 60 SMS, 75 synthetic fact examples,
and 50 synthetic number examples. The fits are calibration diagnostics. The separate runtime and test pools were not used for
these fits. The full [R1 report](mixture-training-results.md) and [R2 report](mixture-training-seed2-results.md) preserve
task retention, confusion counts, and each fit.

## 6. Human-labeled SNLI practice: selected adapter

The matched comparison started both arms from the same weights. Each received 1,512 training presentations over 378 updates.
The control used 504 real topic/spam examples and 1,008 generated logic presentations. The new arm used the same 504 real
examples, 504 generated logic presentations, and 504 SNLI-labeled presentations.

| Version | Correct / 1,152 | Accuracy |
| --- | ---: | ---: |
| Qwen base | 447 | 38.80% |
| More generated practice | 769 | 66.75% |
| Human-labeled SNLI practice | 996 | 86.46% |

The gain over the matched control was 19.70 points (paired 95% group interval +14.32 to +24.48). All nine predefined checks
passed. The SNLI development source was also used in training, so this result is within-source development evidence, not a
held-out reasoning score. The official SNLI test split was not opened.

| Original-order SNLI label | Control correct / 64 | New adapter correct / 64 |
| --- | ---: | ---: |
| Contradiction | 23 | 58 |
| Entailment | 60 | 57 |
| Neutral | 37 | 52 |

On earlier tasks, selected-order accuracy changes were DBpedia −0.57 points, SMS −0.83, generated facts −1.33, and generated
numbers −3.50. Each passed its fixed five-point check.

The number-task interval was −8.25 to +0.75, showing that the gate checked the observed point loss, not the interval bound.
On the original SNLI panel, the new adapter scored 120/128 against the fixed base anchor of 57/128. Intern and Kev scored
58.16% and 76.13% on the balanced panel under different prompts and precision.

Reload checks reproduced both adapters' exact tensors and answers. The new run changed 120 tensors, and an independent
calculation matched the score counts, group means, confusion matrices, checks, and intervals. See the [natural-reasoning
report](natural-reasoning-results.md) and [verification summary](verification/natural-reasoning-summary.json).

## 7. BoolQ and COPA transfer check

This run performed no training. It compared the selected adapter with fresh Qwen base on 32 previously inspected development
questions per task. Each was tested in original and reversed order; original-order results are primary. The interval
resamples paired source groups.

| Task | Base original order | Adapter original order | Difference, 95% interval |
| --- | ---: | ---: | ---: |
| BoolQ reading | 17/32 (53.1%) | 27/32 (84.4%) | +31.3 points (+9.4 to +53.1) |
| COPA cause and effect | 24/32 (75.0%) | 27/32 (84.4%) | +9.4 points (−12.5 to +31.3) |

Across both orders, BoolQ was 39/64 for base and 50/64 for the adapter (+17.2 points, 95% interval +1.6 to +31.3). COPA was
36/64 and 58/64 (+34.4 points, +26.6 to +42.2). Answer changes fell from 21/32 to 4/32 on BoolQ and 28/32 to 6/32 on COPA.

The COPA original-order interval includes losses and gains. These panels had already been inspected and do not establish
broad transfer. See the [transfer report](adapter-transfer-results.md).

## 8. Routing and tool-choice practice: failed gate

The 2026-10-05 study compared the current selected adapter, more existing practice, and new routing/tool-choice practice.
Routing means choosing a support queue from rules. Tool choice means selecting a tool that meets needs and permissions, then
choosing the lowest-cost eligible option. Both new task families were model-scored in this study; there is no corresponding
result on the sealed test rows.

Each new skill has 14 groups. Scores average each question across its fixed answer orders, then average within groups and
weight groups equally. The table shows group means; exact fractions and raw counts are in the verification summary.

| New task | Current adapter | More practice | New practice |
| --- | ---: | ---: | ---: |
| Routing (14 groups) | 21.73% | 28.57% | 38.10% |
| Tool choice (14 groups) | 41.96% | 37.50% | 36.31% |

The equal-weight average was 31.85% for current, 33.04% for more practice, and 37.20% for new practice. The gain over more
practice was 4.17 points, below the fixed minimum of five; its 95% interval was −15.77 to +23.21.

The new recipe lost 1.19 points on tool choice versus more practice and 5.65 versus current. It passed 17 of 21 checks; four
failed: average gain, the average's uncertainty bound, and both tool-choice retention checks. Keep the selected adapter.

Earlier-task scores were also group-weighted.

| Earlier task | Current adapter | More practice | New practice |
| --- | ---: | ---: | ---: |
| DBpedia topic recognition (56 groups) | 97.32% | 98.53% | 98.79% |
| SMS spam (60 groups) | 96.67% | 96.67% | 96.67% |
| SNLI sentence reasoning (64 groups) | 86.46% | 89.50% | 86.63% |
| Synthetic facts (25 groups) | 98.67% | 100.00% | 100.00% |
| Synthetic number selection (25 groups) | 81.25% | 90.00% | 87.88% |
| BoolQ reading (32 groups) | 78.12% | 84.38% | 81.25% |
| COPA cause and effect (32 groups) | 90.62% | 85.94% | 87.50% |

All 14 earlier-skill point checks passed, but some uncertainty ranges still allow losses greater than five points. The new
adapters changed answers on 31/68 routing reorderings and 38/68 tool-choice reorderings, compared with 20/68 and 21/68 for
the current adapter. Changes can help, hurt, or switch between wrong answers.

The four questions with missing or complete tool inputs were all missed across 22 presentations. On two questions where no
tool qualified, new practice scored 7/14; more practice scored 14/14. These small groups suggest cases for a new study, not a
cause.

For example, synthetic number selection was 87.88% by group mean for new practice, while its raw count was 294/364 (80.8%).
The [verification summary](verification/runtime-rule-study-summary.json) has exact counts.

Both trained arms completed 336 updates and 1,344 presentations. All 10,808 planned model passes completed; 120/120 adapter
tensors changed per arm while the base stayed frozen. Reload checks reproduced 32/32 answers with zero score difference.

The arms processed 225,924 and 349,296 training tokens; the new arm used 55% more training text. No new calibration or test
files were used. See the [report](runtime-rule-study-results.md) for all 21 checks, paired intervals, answer-order
breakdowns, and verification details.

## 9. Fresh evaluation on 700 public questions

The unchanged base and selected `snli_mix` adapter were compared without training on 300 HANS, 200 WinoGrande, and 200
ARC-Challenge questions. Each was scored in original order and with answer positions shifted once. Counts below include two
views per question; intervals preserve question or HANS-pattern groups.

| Task | Questions | Base correct / 2 orders | Adapter correct / 2 orders | Change, 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Sentence logic (HANS) | 300 | 304/600 (50.7%) | 385/600 (64.2%) | +13.5 points (−0.2 to +26.7) |
| Commonsense (WinoGrande) | 200 | 197/400 (49.3%) | 206/400 (51.5%) | +2.3 points (−3.0 to +7.3) |
| Science (ARC-Challenge) | 200 | 209/400 (52.3%) | 274/400 (68.5%) | +16.3 points (+10.8 to +21.8) |

Only the science interval excludes zero. WinoGrande is near chance for both models. Original-order accuracy was HANS 154/300
versus 192/300, WinoGrande 97/200 versus 99/200, and ARC 103/200 versus 137/200. Semantic answers changed after order
rotation for HANS 294/300 versus 7/300, WinoGrande 155/200 versus 70/200, and ARC 93/200 versus 41/200. Greater stability did
not make every answer correct.

The adapter scored 285/300 (95.0%) on HANS claims supported by the sentence and 100/300 (33.3%) on unsupported claims. This
suggests an answer imbalance; it does not establish its cause. The full 30-pattern breakdown is in the [fresh-evaluation
report](fresh-eval-results.md) and [verification summary](verification/fresh-eval-summary.json).

Exact-match checks found no overlap with the selected adapter's 1,504 training records. That does not rule out paraphrases or
exposure during base pretraining. One known ARC preview was excluded; other preview overlap is unknown.

This measures the whole selected adapter against its base and does not isolate which training change caused the result. No
comparison with Intern or Kev was run on these questions.

## 10. Targeted HANS/WinoGrande R2: gains with retention losses

The later study compared three training states: the earlier adapter, more earlier practice, and new HANS/WinoGrande practice.
Its reserved HANS and WinoGrande questions are different from the 700 public questions above. Each reserved question had two
answer orders. Group-balanced scores can differ from raw request accuracy.

### Questions reserved for the new skills

Values are group means. HANS has 30 groups and 600 presentations; WinoGrande has 200 groups and 400 presentations.

| Reserved task | Earlier adapter | More earlier practice | New practice |
| --- | ---: | ---: | ---: |
| Target HANS | 63.33% | 71.67% | 88.33% |
| Target WinoGrande | 54.00% | 51.00% | 63.25% |

Against more earlier practice, HANS improved by 16.67 points (95% range +5.17 to +28.67) and WinoGrande by 12.25 points
(+5.75 to +19.00). Against the earlier adapter, the gains were 25.00 points (+7.67 to +41.67) and 9.25 points (+2.50 to
+16.00). These intervals show sample uncertainty, not seed variation.

### Earlier-skill checks

ARC is an earlier-skill retention check in this study, not a reserved target question. Values are group means; group counts
appear in task labels.

| Earlier-skill task | Earlier adapter | More earlier practice | New practice |
| --- | ---: | ---: | ---: |
| ARC science (200 groups) | 67.75% | 66.50% | 62.00% |
| BoolQ reading (32 groups) | 78.12% | 76.56% | 71.88% |
| COPA cause and effect (32 groups) | 90.62% | 85.94% | 71.88% |
| DBpedia topic (56 groups) | 97.45% | 99.62% | 97.07% |
| SMS spam (60 groups) | 96.67% | 97.50% | 97.50% |
| Balanced SNLI (64 groups) | 86.46% | 84.03% | 83.68% |
| Synthetic facts (25 groups) | 98.89% | 98.89% | 100.00% |
| Synthetic numeric (25 groups) | 81.00% | 89.63% | 92.38% |

Five of 24 fixed checks failed, so keep `snli_mix` at update 378. Supported HANS claims fell from 94.67% to 80.33%, a
14.33-point loss; unsupported claims rose from 32.00% to 96.33%.

The other failures were ARC versus the earlier adapter (−5.75 points), BoolQ versus the earlier adapter (−6.25), and COPA
versus more practice (−14.06) and the earlier adapter (−18.75). The fixed limit allowed at most five points of loss. The
gains on the new families do not erase these retention failures.

Exact counts and intervals are in the [targeted report](targeted-training-results.md) and [verification
summary](verification/targeted-training-summary.json).

### Monitoring panels and recovery

These two panels were measured but were not among the 24 acceptance checks.

| Monitoring task | Earlier adapter | More earlier practice | New practice |
| --- | ---: | ---: | ---: |
| HANS evaluation (30 groups) | 64.17% | 69.67% | 90.17% |
| WinoGrande development (200 groups) | 51.25% | 53.25% | 56.75% |

Each trained arm recorded 1,200 updates and 32 exact saved-adapter reload checks with zero score difference before the controller
interruption. One training seed means repeatability across initializations was not measured. Shared HANS templates limit
new-template generalization, and adding both datasets together prevents attributing the effect to either dataset.

The controller stopped before writing a complete success receipt, so its execution status remains failed. A separately
reviewed recovery route admitted saved worker results; subsequent calculations matched the study rules and independent
recount. The V2 change corrected Unicode request hashing after the first result calculation; it did not change study data or
metrics. Recovery made no model calls.

Later file-path and size checks did not produce a fresh remote weight hash, and provider-wide extra attempts remain unknown.
The [targeted report](targeted-training-results.md), [recovery amendment](targeted-r2-recovery-amendment.md), and
[verification summary](verification/targeted-training-summary.json) keep this history.

The report's latest Modal workspace snapshot, dated 2026-10-06, showed $4.53 metered and $0 billed after credits. It is a
delayed workspace total, not total project spend or the cost of this training run.

## Data inventory and held-out boundaries

The repository tracks schemas, manifests, recipes, source notices, synthetic fixtures, results, and verification summaries.
It does not contain the Qwen weights or raw public benchmark records used by the later fresh evaluation.

The routed and tool-choice v1 datasets each contain 112 generated requests: 56 train, 14 development, 14 calibration, and 28
sealed-test requests. Independent checks validated source pins, groups, prompts, and answers. Their development requests were
used in the runtime-rule study; the sealed test rows were not scored.

The test sets are small and do not establish broad performance. See the [routing data report](routing-data-results.md), [tool
data report](tool-data-results.md), and [tool-choice foundation](tool-choice-foundation-results.md).

SNLI preparation includes a separate train-only candidate set of 500 records from 500 source groups: 167 entailment, 167
neutral, and 166 contradiction. It is a data provenance artifact, not a separate model result. The natural-reasoning study
above trained with 500 SNLI-labeled examples.

The official SNLI test member was not opened. Exact overlap checks do not establish that Qwen's pretraining excluded SNLI or
related text. See the [SNLI data report](snli-training-data.md).

The CPU quickstart's nine fixture rows are synthetic records with synthetic logits. They are for checking formats and
evaluator behavior, not model accuracy. Run `uv run reflex validate` and `uv run reflex evaluate` with the fixture as
documented in the [CPU foundation](cpu-foundation.md).

## Training, reload, and speed evidence

The studies used explicit cloud runbooks and recorded task manifests, data and source hashes, training updates, evaluation
counts, and teardown state. LoRA reload checks matched saved tensors and selected answers in the rehearsal, mixture,
natural-reasoning, runtime-rule, transfer, and fresh-evaluation runs where reported. The targeted R2 recovery only rechecked
file paths and sizes after the controller interruption; it did not freshly hash remote weight bytes.

The Qwen compatibility smoke loaded the text model and checked candidate mapping and single-versus-batch parity on three
synthetic prompts. A separate tokenizer check verified all 16 `A`–`P` symbols as single tokens after the newline-ending
`Answer:` suffix at the pinned tokenizer revision. These are compatibility checks, not model quality, calibration,
optimized-kernel, latency, or throughput results. The [smoke report](modal-smoke.md) and [CPU foundation](cpu-foundation.md)
preserve scope.

Forward counts, input-token counts, GPU memory, app lifetime, and workspace billing snapshots are not latency or throughput
measurements. No fixed-hardware comparison has established a speed or per-request cost advantage. The separate 4B evaluation
in the research plan is a possible future study; it has not been run or launched.

## Data terms and release status

Experiments reference SNLI, BoolQ, COPA, HANS, WinoGrande, ARC-Challenge, DBpedia-14, and SMS Spam, along with self-authored
synthetic examples. See the tracked [SNLI notice](../data/notices/SNLI-DATA-NOTICE.txt), [BoolQ
notice](../data/notices/BOOLQ-DATA-NOTICE.txt), [COPA license](licenses/copa.txt), [HANS
notice](licenses/fresh-eval-hans.md), [WinoGrande notice](licenses/fresh-eval-winogrande.md), [ARC
notice](licenses/fresh-eval-arc.md), and [DBpedia/SMS pilot notice](licenses/real-pilot-sources.md).

No model weights are released. There is no top-level code license in the repository. A public model release remains future
work.
