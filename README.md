# Reflex

Reflex is a research project for decisions over options supplied at runtime. The current CPU foundation validates decision records and split lineage, applies a deterministic policy to saved candidate logits, and computes offline evaluation metrics. The CPU tools do not run a language model or train adapters. Separate, explicit Modal commands run a Qwen compatibility smoke and the frozen prior-art comparison; neither is a Reflex training run.

All records and logits checked into `data/fixtures/` are synthetic fixtures. They demonstrate the file formats and evaluator, not model results. Separately, the [2026-10-04 Modal receipt](docs/verification/qwen-modal-smoke.json) contains real Qwen model outputs on synthetic prompts and records passed loading, mapping, and single-versus-batch compatibility checks. A narrow tokenization check on the public Qwen3.5-0.8B-Base tokenizer JSON at revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` found that the newline-ending suffix `\nAnswer:\n` tokenizes all 16 `A`–`P` candidates singly; the trailing-space suffix `\nAnswer: ` does not. The smoke does not establish model quality, calibration, optimized-kernel support, or performance. No Reflex-trained accuracy or benchmark result exists.

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

See the [Modal smoke guide](docs/modal-smoke.md) for the plan-only command, separately guarded one-call remote check, and recorded receipt. The smoke is not an accuracy or performance evaluation.

## Research status

The matched prior-art pilots cover COPA, BoolQ, and SNLI. On COPA, Qwen3.5-0.8B-Base, Intern-Decision-0.8B, and Kev-0.8B scored 22/32, 26/32, and 31/32. On the broader two-task pilot, Intern and Kev tied at 26/32 on BoolQ; Kev led SNLI at 22/32, with fewer observed option-order changes. These are small development measurements of existing models, not Reflex results, a broad ranking, or evidence of generalization; see the [COPA results](docs/copa-baseline-results.md) and [BoolQ/SNLI results and limits](docs/broader-baseline-results.md). BoolQ is a disclosed Kev supervised source and Kev's MultiNLI exposure is related to SNLI; selected-row overlap is unknown, as is base-pretraining overlap. The bounded training-mechanics rehearsal is **in progress**. No adapter has been trained and no Reflex accuracy result exists yet. No account is needed for CPU work; the existing Modal profile is configured. Start with the [research plan](docs/research-plan.md) and [runtime decisions ADR](docs/adr/0001-runtime-decisions.md). Published results for other systems belong to their authors and are not Reflex measurements.
