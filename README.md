# Reflex

Reflex is a research project for decisions over options supplied at runtime. The current CPU foundation validates decision records and split lineage, applies a deterministic policy to saved candidate logits, and computes offline evaluation metrics. The CPU tools do not run a language model or train adapters. Separate, explicit Modal commands run a Qwen compatibility smoke, frozen prior-art comparisons, and a bounded LoRA training-mechanics rehearsal.

All records and logits checked into `data/fixtures/` are synthetic fixtures. They demonstrate the file formats and evaluator, not model results. Separately, the [2026-10-04 Modal receipt](docs/verification/qwen-modal-smoke.json) contains real Qwen model outputs on synthetic prompts and records passed loading, mapping, and single-versus-batch compatibility checks. A narrow tokenization check on the public Qwen3.5-0.8B-Base tokenizer JSON at revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` found that the newline-ending suffix `\nAnswer:\n` tokenizes all 16 `A`–`P` candidates singly; the trailing-space suffix `\nAnswer: ` does not. The smoke does not establish model quality, calibration, optimized-kernel support, or performance. No final sealed-test result or broad benchmark result exists.

## CPU quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Install the locked environment, then run the local fixture commands:

```sh
uv sync --locked --dev

uv run reflex validate \
  --records data/fixtures/records.jsonl \
  --manifest data/fixtures/manifest.json

uv run reflex evaluate \
  --records data/fixtures/records.jsonl \
  --manifest data/fixtures/manifest.json \
  --predictions data/fixtures/predictions.jsonl \
  --output /tmp/reflex-fixture-report.json
```

Validation and evaluation run locally on CPU and do not contact a model service. `evaluate` joins predictions by `record_id` and writes a JSON report with input hashes. See [the CPU foundation guide](docs/cpu-foundation.md) for the formats, metrics, calibration limits, and leakage controls.

See the [Modal smoke guide](docs/modal-smoke.md) for the plan-only command, separately guarded one-call remote check, and recorded receipt. The smoke is not an accuracy or performance evaluation. The [training runbook](docs/training-rehearsal-run.md) defaults to plan-only; its [completed rehearsal results](docs/training-rehearsal-results.md) and canonical receipts record fixture-only training mechanics evidence.

## Research status

The matched prior-art pilots cover COPA, BoolQ, and SNLI. On COPA, Qwen3.5-0.8B-Base, Intern-Decision-0.8B, and Kev-0.8B scored 22/32, 26/32, and 31/32. On the broader two-task pilot, Intern and Kev tied at 26/32 on BoolQ; Kev led SNLI at 22/32, with fewer observed option-order changes. These are small development measurements of existing models, not Reflex results, a broad ranking, or evidence of generalization; see the [COPA results](docs/copa-baseline-results.md) and [BoolQ/SNLI results and limits](docs/broader-baseline-results.md). BoolQ is a disclosed Kev supervised source and Kev's MultiNLI exposure is related to SNLI; selected-row overlap is unknown, as is base-pretraining overlap. The bounded 64-record, self-authored LoRA rehearsal completed after 16 updates with 128/128 fixture presentations correct and exact fresh-reload parity. This verifies training mechanics and memorization on its training fixture only.

The R2 real-data LoRA pilot completed, but its preregistered learning gate failed. DBpedia-14 and SMS Spam macro accuracy rose from 49.35% to 92.98%, while SNLI accuracy fell by 43.75 percentage points and failed the -5-point non-regression requirement. A metadata audit found severe selection skew among the chosen SNLI examples; see the [pilot report](docs/real-pilot-results.md) for evidence and limits. The exact pinned Intern/Kev comparison on the R2 rows is complete; both reference-minus-Qwen-final DBpedia/SMS task-macro intervals include zero, so the report assigns no overall winner ([results](docs/real-pilot-baselines-results.md)).

The separately balanced SNLI diagnostic panel is committed in signed commit `9844a25` with 192 records across 64 source groups ([data](docs/snli-diagnostic-data.md), [protocol](docs/snli-diagnostic-protocol.md)). Its completed R1 run scored 1,152 all-permutation presentations per model: Qwen base 38.80%, the R2 adapter 36.28%, Intern 58.16%, and Kev 76.13%. The paired adapter-minus-base difference was -2.52 percentage points (95% source-group bootstrap interval -4.25 to -0.87). On original-order inputs, the adapter predicted entailment for every record (64/192 correct), so a lower flip count alone is not evidence of useful stability. These are conditional development results; they do not reverse R2's failed gate or establish broad superiority. See the [diagnostic report](docs/snli-diagnostic-results.md). The reviewed synthetic candidate has now been used in the controlled training comparison below.

Both controlled-mixture runs passed all seven checks on the balanced SNLI development panel. Keep their seed results separate:

| Run | Real-only | Synthetic mix | Gain |
| --- | ---: | ---: | ---: |
| R1 | 617/1,152 (53.56%) | 781/1,152 (67.80%) | +14.24 pp |
| R2 | 493/1,152 (42.80%) | 748/1,152 (64.93%) | +22.14 pp |

These two runs do not measure seed uncertainty. Kev scored 877/1,152 (76.13%), ahead of both synthetic-mixture runs. See the separate [R1 results](docs/mixture-training-results.md) and [R2 results](docs/mixture-training-seed2-results.md), with their [R1 verification summary](docs/verification/mixture-training-summary.json) and [R2 verification summary](docs/verification/mixture-training-seed2-summary.json).

The next matched comparison added human-labeled reasoning practice. It scored **996/1,152 (86.46%)**, versus **769/1,152 (66.75%)** for extra synthetic practice. All nine checks passed. These are within-source SNLI development results. See the [result and tradeoffs](docs/natural-reasoning-results.md) and [verification evidence](docs/verification/natural-reasoning-summary.json).

The saved adapter then answered questions excluded from its training:

| Task, original answer order | Fresh base | Saved adapter |
| --- | ---: | ---: |
| Reading comprehension (BoolQ) | 17/32 (53.1%) | **27/32 (84.4%)** |
| Cause and effect (COPA) | 24/32 (75.0%) | **27/32 (84.4%)** |

The reading gain is clearer. The cause-and-effect interval includes both gains and losses. Reversing the choices changed the adapter's answer on 4/32 reading questions and 6/32 cause-and-effect questions, versus 21/32 and 28/32 for the base. These are small, previously inspected development panels. They do not establish broad superiority. See the [transfer results and limits](docs/adapter-transfer-results.md). Next, build and verify a small [routing-data generator](docs/runtime-rule-data-plan.md) before proposing more training.

No account is needed for CPU work; the existing Modal profile is configured. Start with the [research plan](docs/research-plan.md) and [runtime decisions ADR](docs/adr/0001-runtime-decisions.md). Published results for other systems belong to their authors and are not Reflex measurements.
