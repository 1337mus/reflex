# Matched Intern and Kev references for real-pilot-v1

**Protocol version:** 1. Freeze and hash this document, the approved source map, and the
evaluation-only payload contract before either reference model scores a pilot presentation.
This is a bounded comparison of two pinned reference models against the saved final Qwen
adapter from the original [real-data pilot](real-pilot-protocol.md). It does not rerun or
retrain Qwen, change its data, panel, prompts, or learning gate, or define a new pass threshold.

## Question and fixed sample

Measure how the pinned Intern and Kev references score the exact 2,572 presentations already
used by real-pilot-v1. Reuse its prepared records, manifest, recipe, semantic option IDs,
request hashes, and complete fixed panel. Do not select, replace, or rank examples using any
model output. The panel contains the fixed DBpedia-14 and SMS development groups, the SNLI
development groups, and the 116 calibration groups. Calibration labels remain local; the
reference workers receive only requests and presentation metadata, never gold labels or
training records.

Validate the original pilot receipt against its pinned records, manifest, recipe, original
pilot protocol, and reviewed source map. Validate each reference receipt against those same
prepared-data pins and fixed 2,572-presentation panel, plus the frozen reference-protocol pin
and its own reviewed source map. The original receipt does not carry the later reference
protocol pin. The saved original receipt is the sole source of Qwen base/final model and
training provenance. Do not manufacture reference training steps, adapter snapshots,
optimizer data, or any other Qwen evidence to fit the original receipt schema. Use a complete
saved final-adapter output and its validated analysis as the Qwen comparator; if that Qwen
state is missing or failed, do not claim reference-versus-Qwen results. No Qwen rerun is
allowed as part of this comparison.

## Pinned models and inference

| Reference | Immutable model revision | Scoring path |
| --- | --- | --- |
| Intern Decision | `internlm/Intern-Decision-0.8B@85a0cc5a99d67ea8d56dfe98115689212867171d` | Reviewed official `load_scorer()` adapter; FP32 raw candidate logits. |
| Kev | `jaredpalmer/kev-0.8b@9a45d25eb2ab761841196625383fa1dff0e56c1e` | Reviewed official `load_scorer()` adapter; FP32 raw candidate logits. |

Do not fine-tune either reference, alter its official prompt or encoder, or substitute a
mutable revision. Remap scores by semantic option ID before analysis and apply the existing
lexicographically smallest semantic ID tie-break, with ties counted and treated as abstentions.
Preserve each adapter's own digest convention: Intern prompt hashes cover rendered chat bytes;
Kev prompt hashes cover token IDs. Do not pass either through Qwen's prompt-hash validator.
Bind adapter source/provenance and exact uploaded module hashes to each receipt.

These are precision- and encoding-mismatched references. Intern and Kev use their reviewed
FP32 paths; the original Qwen base is BF16 and its saved LoRA adapter is FP32. Prompts and
tokenizers are model-specific. This is a comparison on matched requests and semantic option
orders, not a controlled precision, prompt, tokenizer, latency, or cost comparison.

## Fixed scoring and analysis

Each reference scores all 2,572 presentations exactly once. Intern may use one additional
adapter parity forward and Kev two additional adapter checks. The maximum is 5,147 forwards
across both references, of which 5,144 are scored panel presentations. Record auxiliary
forwards separately; they do not add benchmark rows. Do not retry a model automatically or
replace a failed result. Keep a failed model state and any valid partial rows in the analysis;
never report its incomplete outputs as a complete model result.

Analyze only locally saved, validated receipts. Require exact output membership, stable panel
identities, valid option ordering, finite logits, token bounds, source maps, model revisions,
runtime versions, and provenance for each state. A model-specific failure does not erase or
upgrade the other state's status. A Qwen failure or missing complete original-pilot final
state prevents a Qwen comparison; it does not justify synthesizing replacement metadata.

For Qwen-final, Intern, and Kev, keep `T=1` raw metrics separate from metrics at a separately
fitted temperature for that model. Fit one global temperature per model from only the 116
original-order DBpedia/SMS calibration rows, using the frozen 82-candidate grid and equal task
weight (`1/56` per DBpedia row and `1/60` per SMS row). Freeze the temperature before scoring
development and SNLI metrics. Calibration-pool fitted NLL is a fitting diagnostic, not held-out
performance. If a saved Qwen analysis is supplied, validate it by exact comparison with a
deterministic local recomputation from the validated receipt, using the same frozen records,
grid, task weights, and calibration procedure; then reuse the matching saved analysis. If no
saved analysis is supplied, compute the report locally from the saved Qwen outputs. This
verification performs no model inference, makes no new hyperparameter choice, and introduces
no alternate temperature procedure.

Report by dataset: original-order accuracy, Wilson 95% interval, macro-F1, NLL, summed-class
Brier, ten-bin ECE, risk-coverage and AURC, accuracy by option order, all-orders-correct count,
tie counts, and semantic top-answer flip count/rate. Report an unweighted DBpedia/SMS task-macro
accuracy separately from SNLI. Temperature scaling changes confidence metrics, not non-tied
winners.

Compare Intern and Kev separately against the saved final Qwen adapter, paired by source group
on original-order correctness. Report reference-minus-Qwen-final deltas, both discordant counts, and
the preregistered 2,000-replicate source-group bootstrap with seed `20261005`, stratified by
dataset. Option orders are repeated measurements, never independent examples. Report the
uncertainty results regardless of point estimates. These are small development measurements;
they do not establish statistical significance, overall superiority, unseen-family quality,
or generalization. Preserve regressions and failed attempts. Apply the original Qwen learning
gate only to its original Qwen comparison; do not transfer it to Intern or Kev.

## Data rights and exposure limits

Follow the original protocol and [tracked source notice](licenses/real-pilot-sources.md).
Keep all source-derived records, logits, receipts, and analysis private while the rights,
upstream-component, and SMS privacy reviews remain open. DBpedia's metadata and notices require
attribution and describe CC BY-SA 3.0 plus GFDL terms. UCI currently grants CC BY 4.0 for the
collection; its upstream component corpora were not separately audited and message text has not
had a privacy review. Preserve the intentional ham/spam balance of the existing SMS sample.

Kev discloses DBpedia-14 training exposure; overlap with the selected rows is unknown. Kev also
has related MultiNLI exposure; SNLI row overlap is unknown. Intern, Qwen, and other model
pretraining overlap are not established. Do not claim contamination-free, unseen, or test-set
performance. Do not publish adapters, weights, raw corpus text, or source-derived artifacts.

## Execution boundary

The runner is evaluation-only and plan-only by default. Its payload contains the fixed panel,
never labels or training data. A paid run requires the explicit launch action, existing
personal Modal profile `reflex-personal`, workspace `rajath-61258`, fresh run ID/nonce and
output reservation, no credential environment overrides, and measured source verification
before model load. Public Hub reads are anonymous; models remain on Modal and are never
downloaded to the local workspace.

Two single-use A10 workers may run concurrently, one per reference: 2 physical CPU cores and
16 GiB memory each, maximum two / minimum zero containers, no buffer, no retries, 300-second
startup timeout, and 3,600-second function timeout. Do not create a volume, endpoint, schedule,
or warm pool. These resource and time bounds are not dollar limits; verify account billing
separately. Upload only the reviewed Python sources, protocol files, and explicit evaluation
payload. Never upload the repository root.
