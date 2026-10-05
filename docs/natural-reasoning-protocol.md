# Natural-reasoning mixture protocol

**Status:** Frozen candidate protocol, 2026-10-04. This is a new schema-v2 experiment and
analysis identity (`natural-reasoning-v1`). It replaces the prior mixture protocols for
this run. The former R1 protocol remains available at
[`mixture-training-protocol.md`](mixture-training-protocol.md), SHA-256
`06d3014a45ebeb19500ba2ab002f0595c42b44e561fbb9efc329af0d02a1030c`; the seed-2 protocol
remains at [`mixture-training-seed2-protocol.md`](mixture-training-seed2-protocol.md),
SHA-256 `2cf62fa31f3e52a41ebd632e4883fc6c2f082194a7fa9750b455eb78735c12c8`. They are
historical records, not inherited requirements. The preserved analyzer/runner source commits
for historical reproduction are R1 `2e46d664dc970cd3d6bced19b4f5524fc68d01ae` and R2
`5f7cef988629c682415b5ebf2ed1380fdc90e30b`. The current schema-v2 analyzer intentionally
rejects receipts produced under those older schemas. No run is authorized by this document.

## Objective and interpretation

Compare a synthetic-repeat control (`synthetic_repeat`) with an SNLI-mixture candidate
(`snli_mix`) while keeping initialization, shared examples, update count, and total
presentation count fixed. The control receives extra synthetic examples; the candidate
receives SNLI training examples in those same stream slots. This tests the replacement of a
second synthetic pass with within-source SNLI practice. It does not test whether SNLI data
improves general reasoning or transfers to an independent source.

SNLI is now represented in training and evaluation. Both the original and balanced SNLI
panels are within-source development evidence. They are not unseen-source transfer tests or
sealed tests. The synthetic reasoning tasks are also development evidence. These results
cannot establish broad generalization, benchmark superiority, or release readiness.

## Frozen data, source, and split pins

Use the existing real-pilot, balanced-SNLI, and synthetic-seed-v1-r2 artifacts with the
exact SHA-256 pins already recorded in `experiments/mixture_training_contracts.py`. Add the
private SNLI training candidate with these pins:

| Artifact | SHA-256 |
| --- | --- |
| `data/raw/snli_1.0.zip` | `afb3d70a5af5d8de0d9d81e2637e0fb8c22d1235c2749d83125ca43dab0dbd3e` |
| `data/processed/snli-training-v1/records.jsonl` | `976a0c06f679b9bd989871580b893d32ed593b0912c191ccba56237f7407f2fc` |
| `data/processed/snli-training-v1/manifest.json` | `4be0e00a9588fb7d9077b41c67227ad634737c74910680ad9c24d572c62618b7` |
| `data/processed/snli-training-v1/recipe.json` | `aced7f92bdef65d97fad9fafb2d7f20abef107229bc8e5dea3594969dbd5de0d` |

The source archive is SNLI 1.0. The candidate uses selection seed `20261008`, has 500
records in 500 source groups, and has label quotas of 167 entailment, 167 neutral, and 166
contradiction. The builder excludes complete source-linked training components touched by
SNLI development anchors before selection. It rejects exact normalized pair overlap with
development. No SNLI test member may be read or used.

The candidate builder source pins are:

| Source | SHA-256 |
| --- | --- |
| `src/reflex_decisions/snli_training_source.py` | `a4be159c6c799bf5215ccf5a337a25e34722be92b3c21a7ba6d9a4b8fb01148a` |
| `src/reflex_decisions/snli_training_data.py` | `3106b9ec77bfeb6bbedd2e2768bbc01c4503a6b6f877611b8e9a4c1602b1ae4c` |
| `experiments/prepare_snli_training.py` | `4e2e407aeb049a7f9433f32ae0ce831e9cdf6ce8b1387d5878721deac95db3bb` |

The local preflight must rebuild the candidate from the pinned archive, compare canonical
records, manifest, and recipe bytes with the pinned files, verify every listed digest, and
reject absent or changed pins. Preserve source groups. Reject training/evaluation overlap by
record ID, source group, and exact semantic request. Training payloads contain labels only
for approved training records; evaluation payloads are request-only and contain no answer,
gold label, or solver evidence.

## Paired schedule

Use initialization seed `20261006`. The arms have run-ID suffixes `-syn` and `-snli`.
Each arm receives 1,512 single-example presentations and exactly 378 optimizer updates.
Every update consists of four sequential, unpadded microbatches. The flattened three-stream
schedule repeats its 12-slot real/synthetic/arm-specific pattern across each three-update
window. Save update 189 as unscored evidence. Update 378 is the only checkpoint evaluated or
reloaded.

Build and freeze these deterministic streams:

| Stream | Seed | Epochs and length |
| --- | ---: | --- |
| Shared real | `20261007` | Epoch 0, all 504 records |
| Shared synthetic | `20262007` | Epoch 0, all 500 records, then first 4 from epoch 1 (504 total) |
| Control-specific synthetic | `20262007` | Epoch 1, all 500 records, then first 4 from epoch 2 (504 total) |
| Candidate-specific SNLI | `20263007` | Epoch 0, all 500 records, then first 4 from epoch 1 (504 total) |

Zip the shared real, shared synthetic, and arm-specific streams by position, then flatten
each triple in that order. The first two examples of every triple must match exactly between
arms, including record identity, option order, and gold mapping. The third slot is synthetic
for `synthetic_repeat` and SNLI for `snli_mix`.

`synthetic_repeat` has 504 real and 1,008 synthetic presentations and no SNLI examples.
`snli_mix` has 504 real, 504 synthetic, and 504 SNLI presentations. The union of the two
training sources contains 1,504 unique records: 504 real, 500 synthetic, and 500 SNLI.
With the pinned records and seeds, each arm's shared synthetic stream presents 496 records
once and 4 records twice. The control-specific epoch-1 stream repeats those same four
epoch-1 presentations, and its epoch-2 tail repeats four more records: the control therefore
has 492 synthetic records presented twice and 8 presented three times. The SNLI-specific
stream presents 496 records once and 4 twice. Repeated presentations do not add unique
records. Hash the complete arm schedules, option orders, and gold mappings before launch.

## Model, training, evaluation, and calibration

Use the pinned `Qwen/Qwen3.5-0.8B-Base` checkpoint at revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. Keep the base BF16 and frozen. Use an FP32
LoRA adapter with rank 8, alpha 16, and dropout 0. Train with AdamW at fixed learning rate
`1e-4`, zero weight decay, and gradient clipping at 1.0. This learning rate is a candidate
fixed before training, not a tuned optimum. Do not early-stop, search settings, or select a
checkpoint from evaluation scores.

Preserve the existing label-free evaluation panel exactly at 4,023 presentations per arm:
1,568 DBpedia, 120 SMS, 116 real calibration, 128 original-SNLI, 1,152 balanced-SNLI, 814
synthetic-development, and 125 synthetic-calibration. Preserve its existing 11 data-file
hashes as well as the four SNLI artifact hashes above. Do not score an intermediate
checkpoint.

Raw accuracy is primary. Report original-order accuracy separately. For DBpedia/SMS, report
the fixed selected-order accuracy and original-order result per task. For synthetic data,
report raw accuracy by family and numeric option count. The atomic-fact development panel
has 75 records scored in six orders; the numeric panel has 50 records scored in the fixed
cyclic orders (binary records use both orders; larger option counts use all fixed rotations).
For each synthetic family, compute record accuracy over its selected orders and average
source-group means equally. For balanced SNLI, compute each of 64 source groups' accuracy
over its three records and six orders, then average the 64 group means equally. For every
dataset, first average correctness over selected orders within each record, then average
record accuracies equally within each source group, and finally average source-group means
equally. Report all states separately, require all six development datasets and all three
states, and do not treat orders as independent examples.

Use a paired source-group bootstrap with 2,000 draws, seed `20261007`, sorted group IDs
within each dataset, and type-7 percentile interpolation for 95% intervals. Resample groups
within each dataset and retain paired model predictions. These intervals describe
sample-based uncertainty conditional on one training seed, not training-seed variation.

Fit a separate temperature for the base and each final arm using only the same 241
original-order calibration rows: 56 DBpedia, 60 SMS, 75 synthetic atomic-fact, and 50
synthetic numeric. Weight rows so each of these four source tasks contributes equally to
mean calibration NLL. Use the fixed 82-candidate grid: 81 log-spaced temperatures from
0.05 to 10 plus exact 1.0. No SNLI labels enter temperature fitting. Report temperature,
raw and fitted NLL, Brier score, ECE, and risk-coverage summaries as descriptive confidence
diagnostics only. Calibration or confidence metrics do not determine a gate.

## Predeclared engineering gates

All nine components below must pass for `snli_mix` to advance. `synthetic_repeat` is the
matched control. Differences are exact fractions calculated from source-group means; five
percentage points is `1/20`. Bootstrap intervals are reported for the comparisons, but only
the primary SNLI improvement requires a positive lower bound.

1. The balanced-SNLI all-order group-mean delta (`snli_mix - synthetic_repeat`) is at least
   `1/20`, and its paired 95% lower bound is greater than zero.
2. Balanced-SNLI `snli_mix` accuracy is no more than `1/20` below the pinned base.
3. For DBpedia-14 selected-order accuracy, `snli_mix` is no more than `1/20` below
   `synthetic_repeat`.
4. For SMS selected-order accuracy, `snli_mix` is no more than `1/20` below
   `synthetic_repeat`.
5. For DBpedia-14 original-order accuracy, `snli_mix` is no more than `1/20` below the
   saved R2 final-adapter anchor `50/56`.
6. For SMS original-order accuracy, `snli_mix` is no more than `1/20` below the saved R2
   final-adapter anchor `58/60`.
7. On the original 128-row SNLI panel in original order, `snli_mix` is no more than `1/20`
   below the pinned Qwen-base anchor `57/128`.
8. On synthetic atomic-fact inference, the `snli_mix - synthetic_repeat` equal-source-group
   mean delta is at least `-1/20`.
9. On synthetic numeric selection, the `snli_mix - synthetic_repeat` equal-source-group
   mean delta is at least `-1/20`.

These are fixed engineering thresholds, not claims of statistical superiority. SNLI panels
are within-source development evidence, and synthetic checks cannot substitute for the
primary balanced-SNLI comparison or real-task retention. Preserve every failed gate. Do not
tune a threshold after observing outputs.

## Execution and budget

Run a CPU preflight that validates train splits, exact-solver synthetic labels, group/request
separation, candidate rebuild, all data/protocol/source hashes, source packaging, model and
runtime pins, schedules, panels, and the complete forward/token budget. Freeze the analysis
rule before launch.

Use one single-use A10 worker for shared base evaluation and update-0 adapter creation, with
a 900-second function timeout. Use two concurrent single-use A10 training workers, one per
arm, with 3,600-second timeouts. All workers have a 300-second startup timeout, two physical
CPU cores, and 16 GiB memory. Use zero retries, zero minimum and buffer containers, and a
two-second scale-down window. Before each optimizer is created, each arm independently
verifies the immutable update-0 adapter's file hashes, exact tensor keys/shapes/values, and
canonical digest. A matching seed alone is insufficient. Any failed arm makes the pair
failed; preserve valid partial evidence and do not retry or convert a failed receipt into a
pass.

Keep adapters, logs, logits, progress, receipts, and source-derived artifacts private in the
existing personal Modal volume. Validate each receipt and embedded output hash locally. Verify
remote adapter file hashes and retain that evidence; do not download model weights locally.
Fresh-load update 378 and run 32 fixed parity presentations per arm.

| Phase | Maximum forwards |
| --- | ---: |
| Shared base evaluation: synthetic development and calibration | 939 |
| Training: 378 updates × 4 microbatches × two arms | 3,024 |
| Final evaluation: 4,023 presentations × two arms | 8,046 |
| Reload parity: 32 presentations × two arms | 64 |
| **Pair plus shared base phase** | **12,073** |

The maximum input length is 2,048 tokens per forward. Reject longer inputs instead of
truncating. The total input-token ceiling is 24,725,504 tokens. Report actual totals by phase
and arm. There is no configured dollar ceiling; any paid run needs its own explicit launch
authorization. No run has launched under this protocol.
