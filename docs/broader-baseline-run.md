# Reproduce the BoolQ/SNLI development comparison

Updated 2026-10-04. This runbook reproduces the pinned 64-record development
pilot in `data/baselines/broader-dev-recipe.json`, prints its bounded execution
plan, and shows the explicit launch and local analysis commands. See the
[completed pilot results](broader-baseline-results.md). Read the
[protocol](broader-baseline-protocol.md) and [BoolQ source amendment](broader-baseline-source-amendment.md)
before interpreting or launching the comparison. This is a small development
measurement, not a generalization claim or a Reflex result.

Downloaded corpora and prepared records live under ignored `data/raw/` and
`data/processed/`; do not commit or publish those text files. The pinned recipe,
manifest, notices, protocol and amendment carry source and lineage metadata.

## Environment

From the repository root, install the locked CPU environment:

```sh
uv sync --locked --dev
mkdir -p data/raw data/processed outputs
```

The one-off BoolQ Parquet export below uses a separate ephemeral environment
with PyArrow exactly `25.0.1`; it does not add PyArrow to the project lock.

## Acquire and verify the sources

Download the BoolQ validation Parquet shard from the pinned public mirror
revision and its dataset card:

```sh
curl --fail --location --output data/raw/boolq-validation.parquet \
  'https://huggingface.co/datasets/google/boolq/resolve/35b264d03638db9f4ce671b711558bf7ff0f80d5/data/validation-00000-of-00001.parquet'
curl --fail --location --output data/raw/boolq-dataset-card.md \
  'https://huggingface.co/datasets/google/boolq/resolve/35b264d03638db9f4ce671b711558bf7ff0f80d5/README.md'
```

Download SNLI 1.0 and the Stanford project/license page. The archive URL is
requested without a URL fragment; the fragment in the recipe identifies the
single development member used from that archive.

```sh
curl --fail --location --output data/raw/snli_1.0.zip \
  'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
curl --fail --location --output data/raw/snli-project.html \
  'https://nlp.stanford.edu/projects/snli/'
```

Check the downloaded bytes before using them. A mismatch means stop and
investigate the source; do not silently update the recipe pins.

```sh
cat <<'SHA256' | shasum -a 256 -c -
52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8  data/raw/boolq-validation.parquet
07447668f46c7c1e62e24b650ea2a73f1ab34f2b3759b1587c7b7a81b4bf1cf4  data/raw/boolq-dataset-card.md
afb3d70a5af5d8de0d9d81e2637e0fb8c22d1235c2749d83125ca43dab0dbd3e  data/raw/snli_1.0.zip
9b19a0ac70d0362d90cf59742d803cb93f3c8149d480540e4d211c91b8fdea58  data/raw/snli-project.html
SHA256
```

The reference-page checksums are provenance snapshots pinned in the recipe;
the corpus inputs are the Parquet shard and SNLI archive. If a live reference
page has changed, retain the recorded notice and investigate the change rather
than changing the corpus or protocol as part of a run.

## Export BoolQ and extract only the SNLI development material

Export each Parquet row as compact, sorted-key UTF-8 JSON followed by LF. This
preserves the three fields present in the mirror (`question`, `answer`, and
`passage`); the mirror has no title field.

```sh
uv run --no-project --python 3.12 --with 'pyarrow==25.0.1' python - <<'PY'
import json
from pathlib import Path

import pyarrow.parquet as pq

source = Path("data/raw/boolq-validation.parquet")
output = Path("data/raw/boolq-dev.jsonl")
rows = pq.read_table(source).to_pylist()
if len(rows) != 3270:
    raise SystemExit(f"expected 3270 BoolQ validation rows, got {len(rows)}")
with output.open("w", encoding="utf-8", newline="\n") as stream:
    for row in rows:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        stream.write("\n")
PY
```

Use `unzip -p` for exactly the SNLI development JSONL and README members. This
streams only those members to disk; do not unpack the archive wholesale or read
the train or test members.

```sh
unzip -p data/raw/snli_1.0.zip snli_1.0/snli_1.0_dev.jsonl > data/raw/snli_1.0_dev.jsonl
unzip -p data/raw/snli_1.0.zip snli_1.0/README.txt > data/raw/snli-README.txt
```

Verify the derived source files and the frozen preparation inputs:

```sh
cat <<'SHA256' | shasum -a 256 -c -
c81b6520f1a51a8b96de45420b3d1f78184aee00373c77852132b297f2e567ef  data/raw/boolq-dev.jsonl
9c03faff70182ef086ebfeed2cffbabb5fcc6a84a8b3314decbbb5b01f07f4bf  data/raw/snli_1.0_dev.jsonl
a27e78bf4ba7bc26033fac15a6596dd41c7ff47b4909e739199c59f91cf1bd4e  data/raw/snli-README.txt
6450653a445cc12b21304c267cdf0c8976009a01c6982fb07751b349cd81f68b  data/baselines/broader-dev-recipe.json
f6043407231506a32b490321d5e64f65562604809eade71c8393a1d3380cbf86  docs/broader-baseline-protocol.md
74c38813b882b23b31c214bdc9e4f660dea56c54f8a2c445dd9ba2da97e0da59  docs/broader-baseline-source-amendment.md
SHA256
```

## Prepare and validate the selected records

On a fresh checkout, create the canonical prepared file. The command verifies
all four source files and the approved output digest before writing; it refuses
to overwrite an existing output.

```sh
uv run python -m experiments.prepare_broader_data
```

To reproduce into a fresh scratch path when the canonical ignored output already
exists, use:

```sh
uv run python -m experiments.prepare_broader_data \
  --output /tmp/reflex-broader-dev-validation.jsonl
```

The approved hashes are `5d09ac1c1df2b7c1849e675179cb57306040f473c0ebad60527f54d186ac6845`
for `data/processed/broader-dev-pilot-v1.jsonl` and
`8dbddaec2ceac622ed998cd0c6051472c11bf38d776a895c08ff94ad5968b085` for
`data/baselines/broader-dev-manifest.json`. The preparation step must report the
records digest exactly. The recipe selects 32 source groups per dataset,
without answer balancing, and excludes SNLI rows with gold label `-`.

```sh
cat <<'SHA256' | shasum -a 256 -c -
5d09ac1c1df2b7c1849e675179cb57306040f473c0ebad60527f54d186ac6845  data/processed/broader-dev-pilot-v1.jsonl
8dbddaec2ceac622ed998cd0c6051472c11bf38d776a895c08ff94ad5968b085  data/baselines/broader-dev-manifest.json
SHA256
```

## Inspect the plan and launch explicitly

The default runner invocation is plan-only and does not contact Modal or a
model service:

```sh
uv run python -m experiments.modal_broader
```

Review the printed plan against the protocol before launch: 64 records, 768
scored presentations total, plus the pinned connector verification forwards;
three pinned models; at most three ephemeral A10 containers; zero retries, no
warm pool, endpoint, schedule or volume. The plan also prints exact runtime and
resource bounds. These limits do not set a dollar ceiling.

Keep the protocol, source amendment, runner and inference implementation files
unchanged from plan through analysis. The receipt fingerprints these sources;
analysis rejects a receipt if the current source fingerprints differ.

Only the following command opts into remote inference. It requires the existing
`reflex-personal` Modal profile to resolve to workspace `rajath-61258`, and the
output path must be new. Keep the returned receipt; it contains source
fingerprints, data/protocol hashes, model loading evidence and raw presentation
scores needed by the local analyzer.

```sh
RUN_ID="broader-$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_RECEIPT="outputs/${RUN_ID}.json"

uv run python -m experiments.modal_broader \
  --launch \
  --profile reflex-personal \
  --workspace rajath-61258 \
  --records data/processed/broader-dev-pilot-v1.jsonl \
  --manifest data/baselines/broader-dev-manifest.json \
  --output "$RUN_RECEIPT"
```

The launcher validates the receipt path, inputs, protocol hashes, profile and
workspace before creating remote work. It rejects credential environment
overrides; use the configured Modal profile instead of exporting credentials.
An explicit `--launch` can incur Modal charges.

## Analyze a completed receipt locally

After a passed runner receipt exists, calculate the preregistered metrics on the
local machine. Analysis makes no remote calls. The receipt path is the required
positional argument; the CLI has no `--run` flag. For example, analyze the
canonical completed receipt and print the report to stdout:

```sh
uv run python -m experiments.analyze_broader \
  docs/verification/broader-baselines.json
```

For a new run, choose a new ignored output path or omit `--output` to print the
report to stdout. Set `RUN_ID` to the exact ID used for the runner receipt.

```sh
RUN_ID="broader-your-actual-run-id"
RUN_RECEIPT="outputs/${RUN_ID}.json"
ANALYSIS="outputs/${RUN_ID}-analysis.json"

uv run python -m experiments.analyze_broader "$RUN_RECEIPT" \
  --records data/processed/broader-dev-pilot-v1.jsonl \
  --manifest data/baselines/broader-dev-manifest.json \
  --output "$ANALYSIS"
```

The analyzer verifies source fingerprints, prompt/permutation coverage and
receipt provenance before computing original-order accuracy and its Wilson
interval, plus the preregistered per-dataset metrics and paired comparisons.
Treat option permutations as repeated measurements, not additional independent
questions. Report BoolQ and SNLI separately and retain the protocol's disclosed
training-overlap and independence limits.
