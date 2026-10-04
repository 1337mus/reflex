# One-shot Qwen compatibility smoke

The default command prints a plan and returns. It does not import Modal or Torch, run the Modal CLI, contact a model service, or launch compute. The `uv --with` option supplies the pinned local Modal CLI dependency without changing `uv.lock`:

```sh
UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.cache/python" \
  uv run --locked --with modal==1.6.1 python -m experiments.modal_smoke
```

A paid launch is a separate, explicit command. Use a new output filename; existing files are refused. The parent directory must already exist.

```sh
mkdir -p artifacts
UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.cache/python" \
  uv run --locked --with modal==1.6.1 python -m experiments.modal_smoke \
  --launch --profile reflex-personal --workspace rajath-61258 \
  --output artifacts/qwen-smoke-2026-10-04.json
```

Before importing the Modal SDK or creating a Modal app, the launcher reserves and write-checks the output path, rejects nonempty token and OAuth environment overrides, sets `MODAL_PROFILE`, then runs `sys.executable -m modal token info` with output captured in memory. It requires exactly one `Workspace:` line whose name matches `--workspace`; an optional parenthesized workspace ID can also be supplied in that argument. The token ID and command output are never printed or written to the result. Any check failure after output reservation is saved as a sanitized failed artifact. No credentials are mounted or passed to the remote container.

The remote image uses Debian slim and Python 3.12. It pins Modal locally to 1.6.1, and pins remote `torch==2.14.1+cu130` from the official CUDA 13.0 PyTorch wheel index, `transformers==5.18.0`, and `pydantic==2.13.5`. The tokenizer, safetensors, and Hugging Face Hub packages are resolved transitively and their actual versions are recorded in the artifact. These image pins are not a complete transitive hash lock. The exact CPython 3.12 CUDA 13.0 wheel is listed in the official index; it has not been installed or tested in a Modal image. No model weights are downloaded during image build: anonymous tokenizer and checkpoint downloads happen inside the remote function at the pinned Qwen revision.

The function uploads only `src/reflex_decisions` with `copy=True`. Modal serializes the exact runner function by value with `serialized=True, include_source=False`; it does not mount the repository, caches, or credential directories. The function makes one synchronous remote call, uses one A10, requests and caps CPU at 2 physical cores and memory at 8 GiB, starts with zero warm or buffer containers, sets a 2-second scale-down window, uses `single_use_containers=True`, and sets `retries=0`, a 300-second startup timeout, and a 900-second input timeout. It does not deploy, detach, schedule, map, spawn, mount a volume, or expose an endpoint.

These are resource and call-count controls, not a dollar ceiling. At published A10, CPU, and memory rates, the configured maximum startup and input times plus the 2-second scale-down window estimate to about **$0.42** for one container. This estimate excludes image-build time and other charges. Modal may reschedule platform failures and its SDK can retry internal failures (up to eight) despite `retries=0`; total lifetime starts are not bounded by `max_containers=1`. Modal has not been run against this image, and acceptance of the short scale-down setting and exact wheel remains unverified. Actual usage can exceed the estimate.

The remote check loads `Qwen/Qwen3.5-0.8B-Base` at revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` with BF16 weights, `trust_remote_code=False`, eager attention, and the built-in Transformers key conversion. It checks the expected Qwen text model/config, 24 layers, tied embeddings, absence of meta parameters, and the four public loading diagnostics: `missing_keys`, `unexpected_keys`, `mismatched_keys`, and `error_msgs`. The pinned Transformers 5.18.0 result does not expose a separate `conversion_errors` field. Only the model-declared `model.visual.*` and `mtp.*` unexpected keys are allowed. It forces the Transformers Torch reference fallbacks with `USE_HUB_KERNELS=NO` and `use_kernels=False`, and does not install optional kernel packages. The run records relevant function availability and package versions; it makes no optimized-kernel claim.

Three fixed synthetic requests cover two options, sixteen options, and reversed two-option semantics. The existing `compile_request` checks single-token answer symbols in the complete prompts. Requests are run singly and together in one left-padded batch with explicit attention masks and position IDs, `use_cache=False`, and `logits_to_keep=1`. Candidate logits come from the native final-token output and are scored through the unchanged `score_candidates` function. The artifact contains semantic option IDs, symbol and token mappings, hashes and lengths, single and batch candidate logits/probabilities, semantic selected IDs, raw parity differences, and the top-ID agreement as a separate check.

Parity passes only when maximum absolute candidate-logit difference is at most **0.125** and maximum absolute probability difference is at most **0.02**. Non-finite outputs, loading failures, or mapping inconsistencies fail the artifact and the command exits nonzero. Reversing option order checks the symbol-to-option mapping; it does not require prediction order invariance or a correct answer. This is a compatibility smoke, not a quality, calibration, latency, or benchmark result.

Artifacts are strict JSON, written atomically without replacement under the ignored `artifacts/` directory. A returned remote exception records its stage, type, and up to 2,000 characters of sanitized message; credential-like assignments and bearer/Hugging Face tokens are redacted, while URL userinfo, query strings, and fragments are removed. Local auth and SDK failures remain generic. A platform interruption or timeout may only appear in Modal logs. GPU model loading and inference remain unverified until the explicit remote command completes.
