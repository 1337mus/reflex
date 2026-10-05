# Synthetic LoRA training mechanics rehearsal

Updated 2026-10-04. This protocol defines one bounded, remote-only rehearsal of the
Qwen3.5-0.8B-Base LoRA training path. Its purpose is to test data validation, adapter
construction, gradient flow, persistence, and reload parity. It is a mechanics and
memorization check, not a public benchmark, generalization result, distillation run, or
claim about customer or infrastructure routing quality.

## Data and boundaries

The only training input is the pinned, self-authored 64-record fixture in
`data/training/rehearsal-v1.jsonl`, described by
`data/training/rehearsal-v1-manifest.json`. It contains 32 customer-support examples
(billing versus account) and 32 infrastructure examples (network versus storage). Each
task has eight problem types with four sentence patterns per type. Each exact context is
unique, each gold label is explicit in the record, and each of the sixteen problem-type
groups stays wholly in the train split. The corpus contains no benchmark records or
employer data. COPA, BoolQ, and SNLI remain reserved for later evaluation work.

The launcher re-creates the self-authored recipe, requires byte-for-byte equality with
the checked-in records and manifest, verifies their SHA-256 pins, and runs the split
lineage audit before upload. It sends the two fixture payloads plus a UUID attempt ID
and bounded run/provenance metadata to Modal. Before loading the model, the remote
worker imports each explicitly pinned source module, hashes its `module.__file__`, maps
packaged paths to canonical repository keys, and requires an exact key-and-hash match
with the caller's source map. Receipts keep caller-provided and remotely measured
source fingerprints in separate fields. It does not upload a repository checkout,
local model cache, or credentials.

## Pinned recipe

- Base: `Qwen/Qwen3.5-0.8B-Base`, revision
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`; anonymous Hub access only.
- Image: Python 3.12, Torch 2.14.1+cu130, torchvision 0.29.1+cu130,
  Transformers 5.18.0, and PEFT 0.21.0.
- Adapter: BF16 frozen backbone; rank 8, alpha 16, dropout 0, no bias; target
  modules `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, and `out_proj`.
  Require exactly 60 adapted linear modules, 120 adapter tensors, and 2,015,232
  trainable parameters. Keep LoRA weights FP32; do not cast the PEFT-wrapped model.
- Optimizer: AdamW over trainable parameters only, learning rate `5e-4`, zero weight
  decay, global gradient clip 1.0, seed `20261004`.
- Each update consumes four sequential single-example microbatches without padding.
  Divide each microbatch loss by four, accumulate gradients, clip, then step. Shuffle
  all 64 records each epoch and independently shuffle option order on each presentation;
  map the gold answer by its stable option ID.
- Train only candidate answer-symbol logits at the last prompt position. Use the
  repository's `compile_request` boundary check, `logits_to_keep=1`, and
  `use_cache=False`; gather candidate logits and cast the gathered tensor to FP32 before
  cross-entropy. Do not build sequence labels. The Qwen loader and tokenizer must use
  the same pinned revision, eager attention, `use_kernels=False`, and `token=False`.
- Cap training at 128 optimizer updates (512 training forwards). The maximum including
  128 baseline presentations, four sets of 128 post-training presentations, and 128
  fresh-reload presentations is 1,280 model forwards. Never exceed this cap.

## Diagnostics and success criteria

Score the base model on all 64 original requests and their 64 reversed-option requests.
At updates 16, 32, 64, and 128, repeat the same 128 presentations. Stop at the first
diagnostic with at least 95% overall accuracy, but always train for at least 16 updates,
even when the base already meets that threshold. Report average candidate cross-entropy,
accuracy by task and option order, step losses, selected-answer changes, and candidate
logit changes. These values describe only memorization on this training fixture.

A successful run must show a finite nonzero aggregate LoRA-B gradient, no backbone
gradients, and at least one changed adapter tensor. A high baseline score alone cannot
pass the rehearsal. Record loading diagnostics and the exact target-module/tensor/parameter
inventory. Reject missing or unexpected checkpoint keys and any mismatch with the
pinned inventory.

## Persistence and reload check

Use one uniquely named run directory in the personal Modal Volume
`reflex-rehearsal-artifacts`; refuse an existing run ID. Save adapter-only safetensors,
its PEFT config, and a strict JSON progress receipt at each diagnostic update boundary.
Commit the volume before running each diagnostic. These are inspection snapshots, not
optimizer-resumable checkpoints. Receipts include the UUID attempt ID; data, protocol,
caller-source, remotely measured source, tokenizer, and rendering hashes; model
revision; seed; optimizer settings; completed updates; forward count; training evidence;
and adapter artifact hashes.

If Modal transport or application teardown fails before a receipt is returned, the local
launcher reads only `runs/<run-id>/progress.json` from the Volume. It adopts progress
only after strict JSON, attempt identity, schema, data/model/protocol pins, and both
source maps match. Recovered evidence remains a failed local result, and model or adapter
weights are never recovered.

After the final snapshot, release the training model and optimizer, load a fresh pinned
BF16 base, then load the final adapter unmerged in FP32 and evaluation mode. Disable the
cache. Require exact adapter tensor keys, shapes, and values; finite candidate logits;
identical winners on all 128 presentations; and a maximum absolute candidate-logit
difference no greater than `1e-3`. Persist and commit the final JSON receipt before
returning it. Return only JSON and artifact paths to the local caller, never model or
adapter weights.

## Remote compute and safety

The default launcher only prints a plan and returns. A paid launch requires `--launch`,
an explicit Modal profile, workspace, output path, and safe unique run ID. Before importing
Modal or creating the volume, reserve the output path, verify the exact local data and
protocol hashes, reject credential environment overrides, and confirm the configured
profile resolves to workspace `rajath-61258`. Use public Hub access with `token=False`.

Run one ephemeral A10 function with 2 physical CPU cores, 16 GiB memory, one maximum
container, zero minimum and buffer containers, zero retries, a 300-second startup timeout,
and an 1,800-second function timeout. Do not deploy an endpoint, schedule work, keep a
warm pool, or automatically launch from the default command. These are resource bounds,
not a dollar ceiling. A returned failure receipt is strict JSON, sanitized, and includes
the last completed evidence and adapter paths. No run is authorized by this protocol;
the launcher requires an explicit command-line opt-in.
