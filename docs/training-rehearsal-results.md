# Bounded synthetic LoRA training rehearsal results

The bounded rehearsal `training-rehearsal-2026-10-04-r1` passed on 2026-10-04. It verifies
the pinned training path, adapter updates, persistence, and reload parity on the separate
self-authored 64-record fixture. Every record was used for training, so these numbers measure
fixture memorization only; they are not a benchmark, generalization result, product-quality
finding, or evidence of a Reflex advantage.

## Observed results

The base model scored 101/128 presentations (78.9%) with mean candidate cross-entropy
`0.544683`. At the first permitted diagnostic, after 16 optimizer updates, it scored 128/128
(100%) with mean candidate cross-entropy `0.00002104`. The protocol stops at the first
diagnostic meeting 95% accuracy after at least 16 updates, so later checkpoints were not run.

| Presentations | Base model | Update 16 | Fresh reload |
| --- | ---: | ---: | ---: |
| All | 101/128 | 128/128 | 128/128 |
| Customer support | 42/64 | 64/64 | 64/64 |
| Infrastructure | 59/64 | 64/64 | 64/64 |
| Original option order | 54/64 | 64/64 | 64/64 |
| Reversed option order | 47/64 | 64/64 | 64/64 |

The update changed 27 of the base model's 128 selected answers; the largest candidate-logit
change from base was `10.625`. All 120 adapter tensors changed across 60 modules, with
2,015,232 trainable parameters. LoRA-B gradients were finite and nonzero; the base remained
frozen in BF16 and the adapters remained FP32. The exact tensor inventory and per-update
loss/gradient evidence are in the receipt.

A fresh pinned base with the saved unmerged adapter reproduced all 128 winners. Adapter
keys, shapes, and values matched exactly, and the maximum candidate-logit difference was
`0.0`. The run used 448 of the 1,280 allowed total forwards, including 64 training forwards.
Peak allocated GPU memory was 2,041,713,152 bytes (about 1.90 GiB).

## Provenance and artifacts

The run used `Qwen/Qwen3.5-0.8B-Base` revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, on one ephemeral NVIDIA A10 with BF16 base
weights. The remote environment used Torch `2.14.1+cu130`, torchvision `0.29.1+cu130`,
Transformers `5.18.0`, and PEFT `0.21.0`. The source commit was
[`1e691270cc5816f6df6666be070f4e2e830d1d54`](https://github.com/1337mus/reflex/commit/1e691270cc5816f6df6666be070f4e2e830d1d54);
[CI run 37253123692 passed 184 tests](https://github.com/1337mus/reflex/actions/runs/37253123692).

The four canonical verification artifacts are byte-for-byte copies of the local run evidence:

- Runner receipt: [training-rehearsal.json](verification/training-rehearsal.json), SHA-256
  `92d6afdf6ccebedea26704901fa814d4294bddb945957333e38d871dc3cca073`.
- Persisted progress receipt: [training-rehearsal-progress.json](verification/training-rehearsal-progress.json),
  SHA-256 `72c8889279630a6743266ea4cf0667687ccc7ad12e3347499a724f58559eec80`.
- Volume listing and persistence comparison: [training-rehearsal-storage.json](verification/training-rehearsal-storage.json),
  SHA-256 `8dcb50ca66048232f1f45a0e659a4a2db88d4076288256ff2eb9fb98ebb24ef9`.
- Modal teardown: [training-rehearsal-teardown.json](verification/training-rehearsal-teardown.json),
  SHA-256 `66e2507580e148ca5b66563b104c4f66645bb74cd7c29f4d7928b74b36cff16d`.

The adapter remains in the personal Modal Volume `reflex-rehearsal-artifacts` at
`/artifacts/runs/training-rehearsal-2026-10-04-r1/adapter-update-016`. Its
`adapter_model.safetensors` SHA-256 is
`33e2e55fd56e346e8649fc9ba27afca9bf2bfdd4a0d345f0274072fd725a6190`; the PEFT config SHA-256
is `ca41b120dcbddffa89ee790ff0976212bbe12ccaeb709204525e35e6a5673ae8`. The post-run volume
evidence confirms persisted progress matches the returned receipt. Adapter weights were not
downloaded to this machine.

Modal app `ap-IDGzWaNTLDCftIT1EB8fHI` stopped with zero tasks at 18:53:56 PDT after starting
at 18:52:50 PDT. App lifetime is not model latency or billed runtime. Provider billing was
not verified, so no charge is reported.

The next research step is a preregistered, small real-data multi-task pilot with explicit
held-out datasets or task families and separate train, development, calibration, and sealed
test pools. Do not scale training until its source licenses, leakage checks, metrics, and
acceptance thresholds are frozen. COPA, BoolQ, and SNLI remain development-only.
