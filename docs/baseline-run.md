# Running the COPA development pilot

This guide reproduces the frozen 32-question COPA development pilot. The comparison completed on 2026-10-04; see the [results and interpretation limits](copa-baseline-results.md) before planning another run. Creating a plan or validating local data still does not run inference.

The protocol and exact source, data, and model pins are in [baseline-protocol.md](baseline-protocol.md). Read it before preparing or analyzing a run.

## Prepare and preflight local data

Acquire the pinned source archive and extract only the development XML as described in the protocol. Preparation itself is offline and CPU-only. It verifies the archive, XML, deterministic sample, manifest, and approved hashes; it refuses to replace an existing records file.

```sh
uv run --offline --locked python -m experiments.prepare_baseline_data
```

Run the strict input preflight separately. This validates the exact 32 records, two options per request, development-only manifest, and approved record/manifest hashes without importing Modal or Torch.

```sh
uv run --offline --locked python -c 'from reflex_decisions.baseline_data import verify_prepared_data; manifest, records = verify_prepared_data("data/processed/copa-dev-pilot-v1.jsonl", "data/baselines/copa-dev-manifest.json"); print(f"validated {len(records)} records from {manifest.datasets[0].split} split")'
```

The default runner invocation is plan-only. It does not contact Modal, download weights, or start compute.

```sh
UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.cache/python" \
  uv run --offline --locked python -m experiments.modal_baseline
```

## Explicit paid launch

Choose a unique output filename and create its parent directory first. The command below uses the named `reflex-personal` profile and requires Modal to report the expected `rajath-61258` workspace. Keep token and OAuth override variables unset; the runner rejects them. The local Modal CLI is pinned to 1.6.1 and must already be cached because this command is offline.

Here `uv --offline` only prevents fetching missing Python dependencies. Once launched, the Python process still contacts Modal and the workers download public model assets from Hugging Face.

```sh
mkdir -p artifacts
UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.cache/python" \
  uv run --offline --locked --with modal==1.6.1 python -m experiments.modal_baseline \
  --launch --profile reflex-personal --workspace rajath-61258 \
  --records data/processed/copa-dev-pilot-v1.jsonl \
  --manifest data/baselines/copa-dev-manifest.json \
  --output artifacts/copa-baselines-your-run.json
```

This starts one Modal map with three model jobs, at most three simultaneous A10 containers. Each asks for 2 CPU cores and 16 GiB memory, has a 300-second startup timeout and 900-second function timeout, and uses zero configured application retries, no warm/buffer containers, and single-use containers. Modal may still reschedule platform failures or retry internal failures, so these settings do not cap total starts or spend. There is **no dollar cap**. The full source and runtime pins are recorded in the protocol and receipt.

Successful per-model receipts count 64 scored presentations plus auxiliary parity/probe forwards: Qwen 0, Intern 1, and Kev 2, for 64/65/66 total forwards respectively. Failed models do not claim completed presentation or total-forward counts. The completed run loaded Qwen `T=1.0`, Intern `T=2.747760550703`, and Kev `T=2.3510958125672174`; the [results page](copa-baseline-results.md) reports raw and shipped probability metrics.

Correct-answer labels remain in the local processed file. The remote payload contains only each record ID and unlabeled request; it never includes `answer_id`. Public checkpoint weights are fetched only inside the cloud workers and are not stored on this machine. The `baseline_*.py` adapters are connector code: they translate the request into each model's official input/scoring path. Kev also has a learned LoRA adapter, a small trained weight update over its base model; that is model content, separate from the connector code.

The output file is reserved exclusively before preflight and is never overwritten. Input or protocol failures are written as failed receipts before authentication. Each model receipt retains its own success or failure and any presentations already collected. A top-level run passes only when all three models return all 64 presentations and the Modal app lifecycle exits successfully; a later app-exit exception makes the top-level result failed while preserving model receipts. Analyze only a receipt with top-level `status: passed`.

After the command returns, verify that the named app stopped and has zero tasks. The recorded attempt-3 lifecycle check is in [copa-baseline-teardown.json](verification/copa-baseline-teardown.json): all comparison apps stopped with zero tasks.

```sh
MODAL_PROFILE=reflex-personal UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.cache/python" \
  uv run --offline --locked --with modal==1.6.1 python -m modal app list
```

Check the `reflex-copa-development-baselines` entry in the output. If it is still active or has tasks, stop and investigate before treating the run as finished. The successful attempt-3 receipt and analysis are [copa-baselines.json](verification/copa-baselines.json) and [copa-baselines-analysis.json](verification/copa-baselines-analysis.json); the earlier failures are preserved and explained in the [results page](copa-baseline-results.md).

## Local analysis

Analysis reads the successful receipt and the local answer key; it makes no model or Modal calls. Use a distinct new artifact path.

```sh
uv run --offline --locked python -m experiments.analyze_baselines \
  --run artifacts/copa-baselines-your-run.json \
  --records data/processed/copa-dev-pilot-v1.jsonl \
  --manifest data/baselines/copa-dev-manifest.json \
  --output artifacts/copa-baselines-your-run-analysis.json
```

Canonical accuracy and confidence metrics use the 32 original-order questions. Reversed presentations are paired with those same questions for order-sensitivity checks; they are not extra independent examples. The result is a small development pilot, not an unseen-family or broad-generalization claim. Run provenance records the protocol, records, manifest, and packaged source hashes plus model/runtime pins for reproduction.
