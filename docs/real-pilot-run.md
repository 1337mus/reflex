# Run the first real-data LoRA pilot

This runbook follows version 1 of the [pilot protocol](real-pilot-protocol.md). It covers local
data preparation and verification, the plan-only preflight, one bounded remote run, and CPU
receipt analysis. The commands below were checked against the current parsers; the paid launch
command is an example and has not been run in this session.

## Before a run

1. Read the protocol and [tracked source notice](licenses/real-pilot-sources.md). Confirm
   the prepared recipe contains only the approved DBpedia-14 train source, UCI SMS message
   file, and 128 eligible new SNLI development groups. FinancialPhraseBank must not appear
   in any input.
2. Verify the prepared records, manifest, recipe, source files, protocol, and uploaded module
   hashes against their frozen pins. Expected prepared paths are
   `data/processed/real-pilot-v1.jsonl`,
   `data/pilots/real-pilot-v1-manifest.json`, and
   `data/pilots/real-pilot-v1-recipe.json`. Keep corpus text out of Git.
   If the prepared files are absent and the pinned local source files are available, create
   them with the repository's Python 3.12 environment and the audited DuckDB 1.5.6 package
   from `/private/tmp/reflex-parquet-audit`. The package directory itself is on `PYTHONPATH`.
   The preparation module takes no command-line arguments and refuses to replace any
   different existing prepared artifact.

   ```sh
   PYTHONPATH=/private/tmp/reflex-parquet-audit .venv/bin/python -m experiments.prepare_real_pilot
   ```
3. Confirm the split counts: 504 train, 244 development, and 116 calibration records. Check
   the per-dataset and per-label counts, source-group and request separation, and absence of
   all previously used SNLI groups.
4. Run the repository's targeted CPU contract tests, Ruff, format check, and mypy. Review
   the exact uploaded source map and plan output. The default invocation must remain
   plan-only and must not import Modal, load model weights, or start compute.
5. Before any paid launch, check the current state of the personal Modal account separately.
   The protocol's resource and time limits do not specify or guarantee a dollar cost.

Plan-only preflight (no Modal import or remote compute):

```sh
.venv/bin/python -m experiments.modal_real_pilot
```

The explicit paid launch command below generates a fresh safe run ID and a matching new local
receipt path each time it runs. Keep both unique for every attempt; the remote artifact volume
and local output reservation reject reuse. `MODAL_PROFILE=reflex-personal` selects the existing
personal profile; keep its credentials in the local Modal/uv configuration and do not put
credential values in environment overrides. The runner verifies workspace `rajath-61258` and
rejects credential environment overrides.

```sh
mkdir -p artifacts
RUN_ID="real-pilot-$(uuidgen | tr '[:upper:]' '[:lower:]')"
RECEIPT="artifacts/${RUN_ID}-receipt.json"
MODAL_PROFILE=reflex-personal \
UV_CACHE_DIR="$PWD/.cache/uv" \
UV_PYTHON_INSTALL_DIR="$PWD/.cache/python" \
/Users/raj/.local/bin/uv run --offline --locked --with modal==1.6.1 \
  python -m experiments.modal_real_pilot \
  --launch --profile reflex-personal --workspace rajath-61258 \
  --run-id "$RUN_ID" --output "$RECEIPT"
```

No endpoint, schedule, warm pool, retry, or extra model run is part of this protocol.

## Expected work and saved evidence

| Phase | Work | Maximum forwards |
| --- | --- | ---: |
| Base evaluation | All 2,572 frozen presentations | 2,572 |
| Training | 252 updates × 4 sequential examples | 1,008 |
| Final adapter evaluation | The same 2,572 presentations | 2,572 |
| Reload parity | First 32 presentation IDs in sorted order | 32 |
| **Total** |  | **6,184** |

Before model load, the remote worker must verify its measured module hashes against the
expected source map. It receives labels only for the 504 training examples. Evaluation
payloads contain request text and stable presentation metadata but no labels. Save base
outputs before training. Save and commit the update-126 checkpoint as evidence only; do not
score or reload it. Save and commit the final adapter at update 252 before its evaluation.

Keep the adapter and run evidence in the private `reflex-rehearsal-artifacts` Volume under
`runs/<unique-id>/`. Retain strict JSON progress, the complete base and final semantic
logits, request and prompt hashes, training loss and gradient evidence, adapter files and
hashes, fresh-reload parity details, and final receipt. Return artifact paths and JSON only;
do not download or return model weights. Verify saved paths and hashes after the run.

## Review the result

Accept a run as passed only when the strict receipt matches the run nonce, pinned data,
protocol, model, and both source maps. Confirm it reports exactly 252 updates, 1,008
training forwards, all 2,572 base and final presentation IDs, finite logits for the exact
option IDs, finite nonzero adapter gradients, no backbone gradients, and a changed adapter.
Check the exact adapter tensor inventory after save and reload, matching winners on all 32
parity presentations, and maximum absolute logit difference no greater than `1e-3`.
Persisted progress cannot turn an exception or failed receipt into a passed run. Recovery is
valid only when its strict JSON and unique run nonce match every frozen pin.

Run CPU analysis from returned logits and local labels only. Report original-order accuracy
per task, equal-task DBpedia/SMS macro accuracy, SNLI separately, raw and fitted confidence
metrics, order accuracy, all-orders-correct counts, ties and semantic flips, Wilson
intervals, paired discordant counts, and the stratified 2,000-replicate source-group
bootstrap. Apply the preregistered engineering gate only after reporting the uncertainty
checks. Preserve regressions and failures; do not replace a failed sample or expand the
training budget to make the point estimate pass.

Use the saved local receipt path from the launch command. The analyzer re-verifies all three
prepared data files, rebuilds the fixed 2,572-presentation panel, and exclusively creates its
JSON report; it never launches compute:

```sh
.venv/bin/python -m experiments.analyze_real_pilot \
  --receipt artifacts/run.json \
  --output artifacts/analysis.json
```

Finally, verify the Modal app is stopped with zero running tasks and that the persisted
Volume artifacts match the receipt. Check provider billing separately; app lifetime and
function timeouts do not establish billed cost. Keep source-derived artifacts private while
the rights and upstream-component reviews remain open.
