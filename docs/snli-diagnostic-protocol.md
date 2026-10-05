# Balanced SNLI diagnostic protocol

Updated: 2026-10-04. This protocol freezes a development-only comparison for the
balanced 192-record SNLI panel. It does not alter the original R2 panel, its failed
learning gate, or any prior-art result.

## Data and exposure

Use `data/processed/snli-balanced-v1/records.jsonl`, its manifest, and its source-only
recipe exactly as prepared. The panel contains 64 source groups, with one entailment,
one neutral, and one contradiction record from each group. The deterministic preparation
excludes the source groups used by the earlier 32-group SNLI pilot and the 128-record R2
panel. It reads the pinned SNLI development member only. The 192 records remain
development data; this is not a sealed test set.

For every record, score all six permutations of its original option tuple, in
`itertools.permutations` order. The original semantic order is permutation zero. This
produces 1,152 presentations per model. Remote requests contain presentation and record
identity, source-group identity, request hash, option order, and the request; they contain
no answer, gold label, or other target. Answers stay in the local frozen records.

## Models and scoring

Compare these immutable checkpoints and paths:

| Name | Checkpoint | Revision and path |
| --- | --- | --- |
| `qwen_base` | `Qwen/Qwen3.5-0.8B-Base` | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, BF16 base |
| `qwen_final` | `Qwen/Qwen3.5-0.8B-Base` | same base revision with the exact R2 adapter snapshot |
| `intern` | `internlm/Intern-Decision-0.8B` | `85a0cc5a99d67ea8d56dfe98115689212867171d`, official pinned scorer |
| `kev` | `jaredpalmer/kev-0.8b` | `9a45d25eb2ab761841196625383fa1dff0e56c1e`, official pinned scorer |

The Qwen adapter directory is `/artifacts/runs/real-pilot-2026-10-04-r2/adapter-update-252`.
Before PEFT loads it, verify `adapter_config.json` SHA-256
`fa6fdf55295985c39b5cfd6f6dfb66ab3c941756429cd402f77e1ac455fdea8d` and
`adapter_model.safetensors` SHA-256
`315c23b1517c9590386d22afad5fac2927f97c31f42cdfa73b2c921494266fd8`. Require exact
loaded tensor equality, the default adapter active and unmerged, 60 modules, 120 tensors,
2,015,232 FP32 adapter parameters, and a fully frozen model during inference.

Use the pinned reference runner image: Modal 1.6.1, Python 3.12, Torch 2.14.1+cu130,
Transformers 5.18.0, PEFT 0.21.0, and the package versions recorded in each receipt.
All scoring runs in evaluation and inference mode. Preserve official Intern and Kev
scoring and provenance unchanged. Prompt hashes cover rendered UTF-8 bytes for Qwen and
Intern and the canonical token-ID array for Kev. Reject inputs above 2,048 tokens. Store
three finite raw logits in request option order and choose the lexicographically smallest
semantic option ID on an exact maximum tie; report ties separately.

## Execution bounds

Launch three concurrent, single-use A10 containers: Qwen scores base then final in one
worker; Intern and Kev each use one worker. Each worker receives the same request-only
payload and returns only its own model state. Hard limits are two physical CPU cores,
16 GiB memory, 300-second startup, 1,800-second function timeout, zero retries, zero
minimum and buffer containers, and a two-second scale-down window. The maximum is 4,608
scored forwards plus three official auxiliary forwards, 4,611 total. No endpoint, warm
pool, schedule, local model load, or local model download is part of this run. Launch is
explicit and must use the fixed personal Modal profile and workspace. The rehearsal
artifact volume, when mounted, is read-only.

Before model loading, validate the exact request-only payload, all local data/protocol
pins, the complete source allowlist and remote source hashes, and runtime versions. Every
result binds the run UUID, separate nonce, data/protocol/source pins, model ID and
revision, runtime versions, measured source hashes, scoring provenance, and observed
forward counts. Preserve valid complete worker results if another worker or the Modal
lifecycle fails. A failed model may retain only a validated prefix of the 1,152 requests;
it is never treated as a complete evaluation.

## Analysis and interpretation

Analyze saved receipts locally and independently validate them before scoring results.
The primary metric is accuracy over all 1,152 presentations. Report original-order
accuracy as secondary; per-class recall and confusion over all orders; original-order
confusion; exact-tie counts; per-record semantic-flip and all-six-orders-correct counts;
and equal-weight source-group averages. Report unique-maximum coverage and accuracy
among uniquely decided presentations alongside deterministic tie-broken accuracy.

Use 2,000 paired source-group bootstrap draws with seed `20261006`. Resample the 64
groups, preserving all three records and all six option orders for each selected group.
Report 95% percentile intervals for each complete model's all-order accuracy and for
paired `qwen_final` minus `qwen_base`, `intern`, and `kev` differences. Permuted rows are
repeated measurements, not independent examples. Do not use row-level Wilson intervals
as group-level uncertainty.

This is conditional evidence on a balanced development sample, with possible task-schema
familiarity and pretraining exposure. The reference implementations use different prompt
and tokenizer paths. The panel differs from R2, so the comparison cannot isolate the
effect of balancing from the effect of changing examples. It establishes no broad or
unseen-data superiority and does not create a new training or acceptance gate.
