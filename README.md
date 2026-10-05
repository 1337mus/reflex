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

The first matched prior-art run compared Qwen3.5-0.8B-Base, Intern-Decision-0.8B, and Kev-0.8B on 32 COPA development questions: 22/32, 26/32, and 31/32 correct in original option order. This is a small development pilot for existing models, not a Reflex result or a broad generalization claim; see the [full results and limits](docs/copa-baseline-results.md). No Reflex-trained adapter exists. The next evidence step is a preregistered expansion to two additional task types with 2–3 options and pinned source, license, and group/split lineage, before competitive training. No account is needed for CPU work, and no new account is needed for that study; the existing Modal profile is configured. Hugging Face can wait until checkpoint uploads or gated/private resources are needed. Start with the [research plan](docs/research-plan.md) and [runtime decisions ADR](docs/adr/0001-runtime-decisions.md). Published results for other systems belong to their authors and are not Reflex measurements.
