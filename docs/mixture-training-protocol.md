# Controlled-mixture training protocol

**Status:** Frozen before training, 2026-10-04. This protocol defines the controlled
experiment after the balanced-SNLI diagnostic. R1 results are recorded separately in the
[diagnostic report](snli-diagnostic-results.md). Implementation review and exact-source CI
must pass before the orchestrator launches the user-authorized run. This document contains
no training results. Preserve the original R2 learning-gate failure.

## Objective and comparison

Compare two Qwen LoRA training arms at the same optimizer-update and total training-forward
budget. The `synthetic_mix` arm replaces **all 504 real presentations in the control's
second pass** with exact-solver synthetic presentations. Those 504 replacements are half
of the 1,008 total training presentations; both arms share the same first 504 real
presentations. This tests replacement under a fixed budget; it does not estimate the effect
of simply adding data, because real-data exposure differs between arms. Input-token
exposure may differ and must be measured.

The objective is to retain the useful DBpedia/SMS learning from the earlier real-data
pilot while improving transfer to the balanced SNLI panel and learning across synthetic
task families and option counts. A synthetic-only improvement does not establish transfer
or satisfy real-task retention. Comparisons with the prior R2 adapter also change the
learning rate and are historical contrasts, not isolated causal estimates.

## Data and split boundaries

Use the existing frozen real-pilot records, manifests, and recipes; the 192-record,
64-source-group balanced SNLI development panel and its separate protocol; and the reviewed
750-record synthetic candidate. Keep their lineage manifests separate. Preserve original
`source_group_id` values through schedules, scoring, analysis, and resampling.

| Source | Training | Evaluation per arm | Calibration | Use |
| --- | ---: | ---: | ---: | --- |
| DBpedia-14 and SMS real pilot | 504 train rows: 252 from each task | 1,568 DBpedia and 120 SMS selected-order presentations | 116 original-order rows: 56 DBpedia, 60 SMS | Real-task learning and retention |
| Original R2 SNLI panel | None | 128 original-order presentations | None | Preserve historical panel behavior; do not fit on SNLI labels |
| Balanced SNLI diagnostic | None | 1,152 presentations: 192 records × six orders | None | Primary transfer comparison; 64 source groups, with dependence within each group |
| Synthetic candidate | 500 train rows: 300 atomic-fact, 200 numeric | 814 development presentations | 125 original-order rows | Direct learning by family and option count |

The synthetic development panel has 75 three-option atomic-fact records and 50 numeric
records: 14 binary, 12 four-option, 12 eight-option, and 12 sixteen-option records. Score
all permutations for two- and three-option records. For four, eight, and sixteen options,
score only the fixed cyclic rotations, beginning with the original order; call these
selected-order means, not exhaustive all-order performance. Synthetic calibration remains
original-order only. Revalidate all synthetic training labels with the exact deterministic
solvers. Do not show benchmark, development, calibration, or test examples to a teacher.

Training payloads may contain labels only for their approved train rows. Evaluation
payloads are request-only and must contain no answer, gold label, or solver evidence. Reject
train/evaluation overlap by record ID, source group, and exact semantic request. There is no
sealed-test evaluation in this candidate.

## Paired training schedule

The canonical arm names are `real_only` and `synthetic_mix`.

| Arm | Shared real slots | Arm-specific slots | Total presentations |
| --- | --- | --- | ---: |
| `real_only` | One seeded pass over all 504 real train rows | A second seeded pass over those same 504 real rows | 1,008 |
| `synthetic_mix` | The identical seeded pass over all 504 real train rows | All 504 second-pass real slots replaced by one shuffled pass over 500 synthetic train rows plus four declared repeats | 1,008 |

Use schedule seed `20261007` and initialization seed `20261005`. Construct the shared real
presentations once with `epoch_examples(real_train, seed=20261007, epoch=0)`. Use epoch 1
with the same schedule seed for the real-only replacement pass. Construct the synthetic
replacement from `epoch_examples(synthetic_train, seed=20262007, epoch=0)` followed by its
first four examples from epoch 1 with that same synthetic schedule seed. The four repeats
are repeated exposure, not new unique records. Freeze all example and option orders and the
resulting schedule hashes before launch.

Interleave shared and arm-specific examples in alternating slots in both arms. The shared
real record, semantic option order, and gold-option index must match at every shared slot.
Each four-example update therefore has two shared-real and two arm-specific single-example
microbatches. Real-only sees every real training record twice; synthetic_mix sees each real
record once, each of the 500 synthetic records once, and four synthetic records twice.
This replaces the full second real pass, or half of each arm's 1,008 presentations. Report
input-token totals by arm because equal forward counts do not imply equal token exposure.

Use the pinned `Qwen/Qwen3.5-0.8B-Base` checkpoint at revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. Keep the base BF16 and frozen. Use an FP32 LoRA
adapter with rank 8, alpha 16, and dropout 0. Train with AdamW at fixed learning rate
`1e-4`, zero weight decay, and gradient clipping at 1.0. The learning rate is a candidate
fixed before training, not a tuned optimum. Run exactly 252 updates, four sequential
unpadded microbatches per update, and 1,008 training forwards per arm. Do not early-stop,
search settings, or choose a checkpoint from evaluation scores.

## Shared initialization and execution phases

1. **CPU preflight.** Validate train splits, exact-solver labels, group/request separation,
   schedule and panel counts, all data/protocol/source hashes, source packaging, runtime and
   model pins, Modal profile/workspace, and the complete forward/token budget. Freeze the
   analysis rule and candidate gates below before launch.
2. **Shared base and initialization worker.** Before training, use one single-use A10 worker
   to score the 814 synthetic development and 125 synthetic calibration presentations
   with the pinned base (939 forwards total) and create one FP32 rank-8 adapter from
   initialization seed `20261005`. Existing base outputs for other panels may be reused
   only after exact model, revision, tokenizer, prompt/rendering, runtime, and panel identity
   checks. Save the immutable update-0 snapshot and commit it to the private artifact volume
   before starting either arm.
3. **Paired training.** After the update-0 snapshot is durable, run two bounded A10 training
   workers concurrently, one per arm, with separate run IDs and artifact paths. Each worker
   loads a fresh pinned BF16 base and the same update-0 adapter. Before constructing its
   optimizer, each worker independently verifies the saved file hashes and exact tensor
   keys, shapes, values, and canonical digest against that immutable update-0 artifact. An
   arm may train only after its own check passes; a matching seed alone is insufficient and
   no joint readiness barrier is required. Any failed arm makes the paired experiment
   failed, even if its sibling completes. Allow the sibling to finish for diagnostic
   evidence, but do not report a paired pass or evaluate the primary contrast from an
   incomplete pair.
4. **Snapshots and final scoring.** Save update-126 and update-252 adapter snapshots. The
   update-126 snapshot is unscored evidence only. Final update 252 is selected in advance
   and is the only checkpoint to score or reload. Fresh-load the final snapshot and run 32
   fixed parity presentations per arm before marking that arm passed.

Keep adapter snapshots, logs, logits, progress, receipts, and source-derived artifacts
private in the existing personal Modal volume. Use unique arm-specific run directories.
Validate each receipt and embedded output hash locally before analysis. Verify adapter
file hashes on Modal and retain that evidence; do not download model weights to the local
computer. Preserve failures and valid partial evidence; do not retry or turn a failed
receipt into a passed run.

## Forward budget

| Phase | Work | Maximum forwards |
| --- | --- | ---: |
| Shared base evaluation | Synthetic development and calibration panels | 939 |
| Each arm: training | 252 updates × four microbatches | 1,008 |
| Each arm: final evaluation | Fixed panel below | 4,023 |
| Each arm: reload parity | 32 fixed presentations | 32 |
| Each arm total |  | 5,063 |
| **Pair plus shared base phase** | 2 × 5,063 + 939 | **11,065** |

The 4,023 final-evaluation forwards per arm are 1,568 DBpedia, 120 SMS, 116 real
calibration, 128 original-SNLI, 1,152 balanced-SNLI, 814 synthetic-development, and 125
synthetic-calibration presentations. Do not score intermediate checkpoints.

## Modal and token limits

These limits were fixed before any candidate training outputs:

| Phase | Workers | GPU | Function timeout |
| --- | ---: | --- | ---: |
| Shared base evaluation and update-0 creation | 1 | A10 | 900 seconds |
| Paired training | 2 concurrent, one per arm | A10 each | 3,600 seconds each |

All workers have a 300-second startup timeout, two physical CPU cores, and 16 GiB memory.
Use zero retries, zero minimum and buffer containers, single-use workers, and a two-second
scale-down window. The request limit is 2,048 input tokens per forward; reject longer
inputs instead of truncating. At the 11,065-forward maximum, the total input-token ceiling
is 22,661,120 tokens. Report actual input-token totals by phase and arm. There is no
configured dollar ceiling; any paid run still requires an explicit launch decision.

## Metrics, calibration, and candidate decision rule

Raw accuracy is primary. Report original-order accuracy separately. For DBpedia/SMS, report
the fixed selected-order accuracy and the original-order result per task. For synthetic data,
report raw accuracy by family and numeric option count, with the selected-order semantics
above. For balanced SNLI, compute each source group's accuracy over its three records and
six orders, then average the 64 group means equally. Report the base, `real_only`, and
`synthetic_mix` values; the primary transfer delta is `synthetic_mix - real_only`.

Use a paired source-group bootstrap with 2,000 draws, seed `20261007`, sorted source-group
IDs within each dataset, and type-7 percentile
interpolation to report 95% intervals for the primary balanced-SNLI delta and the required
retention comparisons. Resample groups within each dataset and retain paired model
predictions. These intervals describe evaluation-sample uncertainty conditional on one
training seed; they do not measure training-seed variability. Do not treat repeated option
orders as independent examples.

Fit a separate temperature for the base and each final arm using only the 241 original-order
calibration rows: 56 DBpedia, 60 SMS, 75 synthetic atomic-fact, and 50 synthetic numeric.
Weight rows so each of these four source tasks contributes equally to mean calibration NLL.
Use the same fixed 82-candidate grid as the R2 analysis: 81 log-spaced temperatures from
0.05 to 10 plus exact 1.0. No SNLI labels enter temperature fitting. Report temperature,
raw and fitted NLL, Brier score, ECE, and risk-coverage summaries as descriptive confidence
diagnostics only. Calibration or confidence metrics do not determine the training gate.

The following engineering gates are fixed before training. All must pass for the
`synthetic_mix` arm to advance; `real_only` is the diagnostic control:

1. The balanced-SNLI raw all-order group-mean delta for `synthetic_mix - real_only` is at
   least +5 percentage points, and its paired 95% lower bound is greater than zero.
2. Balanced-SNLI `synthetic_mix` accuracy is no more than 5 points below the pinned base.
3. For each real task separately, `synthetic_mix` selected-order accuracy is no more than
   5 points below `real_only`.
4. For each real task, `synthetic_mix` original-order accuracy is no more than 5 points below the
   saved R2 final-adapter anchor: DBpedia-14 50/56 (89.3%) and SMS Spam 58/60 (96.7%).
5. **Original-SNLI retention safeguard:** on the original 128-row R2 panel in original
   order, `synthetic_mix` accuracy is no more than 5 points below the saved Qwen-base anchor of
   57/128 (44.53%), giving a minimum of 39.53%. This is separate from the balanced-SNLI
   transfer delta and its base-retention safeguard above.

These are engineering thresholds; they do not establish broad statistical superiority. Preserve any failed
gate and do not adjust it after seeing results. Synthetic performance cannot substitute for
the primary `synthetic_mix - real_only` balanced-SNLI transfer gate or per-task retention.
The original-SNLI check is an additional advancement safeguard, not a substitute for the
balanced-SNLI safeguards. If the transfer gate passes, repeat with a second training seed
before broader claims or release decisions.

## Interpretation limits and launch verification

Synthetic atomic-fact inference is related to SNLI reasoning. SNLI may be an unseen source
dataset for this training mixture, but it is not a clean unseen task family. The balanced
panel is development data, not a sealed test. The candidate cannot establish broad
generalization, benchmark superiority, or contamination-free performance. One training seed
does not establish training stability.

Before launch, verify the pinned data and source hashes, schedules and panels against this
protocol. Run the CPU schedule/contract and solver-label checks, verify that the total
budget is at most 11,065 forwards and 22,661,120 input tokens, complete strong review, and
pass exact-source CI. Preserve partial evidence and original failures. Any failed arm
makes the pair fail, including a lifecycle failure after recovery. No automatic retries.
No training run had launched when this protocol was frozen.
