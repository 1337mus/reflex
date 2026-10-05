# Controlled-mixture training results (R2)

On the balanced SNLI development panel, the synthetic-mixture arm scored 748/1,152 (64.93%),
compared with 493/1,152 (42.80%) for its matched real-only control. All seven predeclared R2 checks
passed. R1 and R2 are separate seed results; neither measures training-seed uncertainty or
establishes release readiness.

## Balanced SNLI

The panel has 192 questions. Each question is scored in six option orders, for 1,152 presentations
total. The 192 questions come from 64 source groups, which are the unit resampled for the paired
interval.

| Initialization | Real-only | Synthetic mix | Gain |
| --- | ---: | ---: | ---: |
| R1 (`20261005`) | 617/1,152 (53.56%) | 781/1,152 (67.80%) | +14.24 pp |
| R2 (`20261006`) | 493/1,152 (42.80%) | 748/1,152 (64.93%) | +22.14 pp |

For R2, the paired 95% interval for the gain is **+18.49 to +25.26 percentage points**. It reflects
sampling uncertainty across the 64 groups for this initialization, not variation across training
seeds. Keep the R1 and R2 results separate; do not pool them.

Historical reference scores on this same panel were 670/1,152 (58.16%) for Intern-Decision-0.8B and
877/1,152 (76.13%) for Kev-0.8B. These are from the earlier [balanced SNLI
diagnostic](snli-diagnostic-results.md), not arms in R2. Kev remains ahead of both R2 arms.

On the mixture's 192 original-order presentations, performance varied by label:

| Gold label | Correct |
| --- | ---: |
| Claim conflicts (contradiction) | 44/64 |
| Claim follows (entailment) | 61/64 |
| Not enough information (neutral) | 15/64 |

Neutral remains the main weakness: 29 of 64 neutral claims were labeled “claim follows” and 20 were
labeled “claim conflicts.” The exact confusion matrices for all states are retained in the
verification JSON.

## Topic, spam, and original-SNLI checks

| Evaluation | R2 real-only | R2 synthetic mix | Historical anchor |
| --- | ---: | ---: | ---: |
| DBpedia-14, selected orders | 1,537/1,568 | 1,518/1,568 | — |
| DBpedia-14, original order | 56/56 | 54/56 | Saved R2 adapter: 50/56 |
| SMS Spam, selected orders | 115/120 | 116/120 | — |
| SMS Spam, original order | 58/60 | 58/60 | Saved R2 adapter: 58/60 |
| Original SNLI panel, original order | 4/128 | 97/128 | Qwen base: 57/128 |

All seven checks passed. Real-task retention checks compare the observed accuracy differences with
fixed five-point limits; they do not require the interval's lower bound to clear that limit. Exact
values and intervals for every check are in the [verification
summary](verification/mixture-training-seed2-summary.json).

This result does not reverse the earlier failed real-data R2 pilot. Its final adapter scored 1/128
on the original SNLI panel against the 57/128 base anchor; see the [pilot
report](real-pilot-results.md). That pilot used a different training setup, so the contrast does not
isolate the effect of synthetic examples.

## Run and evidence

R2 changed the initialization seed to `20261006`; the data, schedules, training settings, and gates
matched R1. Each arm received 1,008 presentations. The mixture replaced 504 real presentations with
500 checked synthetic examples plus four repeats.

The run used 11,065 forwards and 2,409,145 input tokens. Remote verification passed for all 15 saved
files, and both Modal apps stopped with zero running tasks. The workspace billing snapshot was
**$1.82 metered, $0 billed after credits**; it is lagging and cannot be attributed to this run.
Exact counts, hashes, billing, and all seven gates are in the [verification
summary](verification/mixture-training-seed2-summary.json).

The frozen [R2 protocol](mixture-training-seed2-protocol.md) incorporates the original [R1
protocol](mixture-training-protocol.md). Source commit
[`5f7cef988629c682415b5ebf2ed1380fdc90e30b`](https://github.com/1337mus/reflex/commit/5f7cef988629c682415b5ebf2ed1380fdc90e30b)
passed [CI run 37270398048](https://github.com/1337mus/reflex/actions/runs/37270398048). Local run
receipts and teardown evidence are under `artifacts/mixture-2026-10-04-r2-*.json`; independent
recounts and root agreement are under `.context/mixture-seed2-*.json`.

Synthetic atomic-fact inference is related to SNLI reasoning, and this is development data rather
than a sealed test. Two successful initializations do not establish broad generalization. See the
separate [R1 report](mixture-training-results.md); preserve both results without pooling.
