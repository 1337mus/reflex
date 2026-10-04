# Reflex

Reflex is a research project for decisions over options supplied at runtime. The current CPU foundation validates decision records and split lineage, applies a deterministic policy to saved candidate logits, and computes offline evaluation metrics. It does not run a language model or train adapters.

All checked-in records and logits are synthetic fixtures. They demonstrate the file formats and evaluator only; they are not model results. A narrow tokenization check on the public Qwen3.5-0.8B-Base tokenizer JSON at revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` found that the newline-ending suffix `\nAnswer:\n` tokenizes all 16 `A`–`P` candidates singly; the trailing-space suffix `\nAnswer: ` does not. This does not establish end-to-end Qwen compatibility: weight loading, inference, and batching remain unverified. There are no Reflex model accuracy or performance results.

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

## Research status

The research question is whether runtime-defined decisions can deliver useful quality, robustness, or selective-risk improvements over existing decision systems. That remains unproven. The current proposal is English, single-label decisions over 2–16 options; it is a research scope, not a model capability. The first experiment compares existing decision models before Reflex training. No account is needed for CPU work; set up Modal before the first remote baseline job. Hugging Face can wait until checkpoint uploads or gated/private resources are needed. Start with the [research plan](docs/research-plan.md) and [runtime decisions ADR](docs/adr/0001-runtime-decisions.md). Published results for other systems belong to their authors and are not Reflex measurements.
