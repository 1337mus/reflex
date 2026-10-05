# First supervised real-data LoRA pilot

**Protocol version:** 1. Review this document with its tracked source notice and prepared
recipe, then commit and pin their hashes before model inference. This protocol covers one
private research run comparing the pinned Qwen base with one LoRA adapter. It is a bounded
feasibility pilot, not a product-quality, broad benchmark, or unseen-data claim.

## Data, rights, and limits

Use only the locally pinned sources in the prepared recipe. Raw and processed corpus text
stays out of Git. The exact record, manifest, recipe, protocol, and source-module hashes
must be checked before any model loads.

| Source | Use | Rights and exposure limits |
| --- | --- | --- |
| DBpedia-14, `fancyzhx/dbpedia_14` at `9abd46cf7fc8b4c64290f26993c540b92aa145ac` | Custom train, development, and calibration groups carved from the official train file. The pinned train Parquet SHA-256 is `0640e4664a99cc94c47db1d7b2e01c14455d5bbecb8183ad1f93bde59f3f28ee`. | The metadata says CC BY-SA 3.0; the card also cites the GNU Free Documentation License. Preserve attribution and notices. Do not publish source-derived records, adapters, or weights until the ShareAlike/GFDL obligations are reviewed. Kev discloses DBpedia-14 training exposure; selected-row overlap is unknown. |
| UCI SMS Spam Collection, dataset 228 / DOI `10.24432/C5CC84` | Custom train, development, and calibration groups from `SMSSpamCollection`. The acquired archive SHA-256 is `1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3`; the extracted message member SHA-256 is `7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d`. | UCI's current page grants CC BY 4.0. This private research use was adjudicated under that dataset-specific grant. Preserve attribution and the archive README's warranty, liability, and indemnity notices. The upstream component corpora were not separately audited, and message text has not had a privacy review. Keep records and derivatives private pending separate review. |
| SNLI 1.0 official development split | 128 additional, source-grouped development examples. The earlier 32 selected development groups are excluded. | Development evidence only, not a sealed test. Earlier work scored other groups from this same split. Kev has related MultiNLI exposure; Qwen pretraining overlap is unknown. Do not claim unseen or contamination-free data. Preserve source notices and the CC BY-SA 4.0 attribution recorded in the tracked source notice. |

FinancialPhraseBank is rejected and must not enter this pilot. The DBpedia official test
Parquet was not acquired. The downloaded SNLI distribution ZIP contains train and test
members, but this pilot extracts, parses, and scores only the development member; no test
member is extracted or evaluated. One DBpedia example visibly marked as a test example
appeared in the upstream dataset card during rights review. It was not copied, selected, or
sent to a model; preserve this exposure note for any later test protocol.

Public citations, acquired-byte hashes, attributions, and use limits are recorded in the
[tracked source notice](licenses/real-pilot-sources.md). The committed
[prepared recipe](../data/pilots/real-pilot-v1-recipe.json) must record the selected groups
and exact source and record hashes. Exact duplicate grouping does not establish semantic
independence. Near-paraphrase overlap, upstream SMS component lineage, and model pretraining
overlap remain unmeasured.

## Frozen allocation

Group records before ranking or splitting. For DBpedia and SMS, normalize text with Unicode
NFKC, case folding, and whitespace collapsing. Connect DBpedia rows when either normalized
title or normalized content matches; discard connected groups with conflicting labels.
Identical normalized SMS messages form one group. For SNLI, follow the existing parser:
case-fold and collapse whitespace without NFKC; connect rows sharing `captionID` or
normalized premise, deduplicate exact premise-hypothesis pairs, drop rows without a consensus
label, and exclude all groups selected in the earlier 32-record SNLI development sample.

For every source, rank groups using
`SHA256("reflex-real-pilot-v1:20261005:{source_id}:{group_id}")`, ascending by digest and
then group ID. Use the canonical source ID shared by its split datasets, never a
split-specific dataset ID. For DBpedia and SMS, use the lowest native row index as each
eligible homogeneous group's representative. Allocate by native label to the class counts
below, taking train, development, then calibration groups in hash order. For SNLI, rank
remaining groups without label balancing and select the first 128; use the lexicographically
lowest native `pairID` as each group's representative. No model output, prompt length, or
evaluation performance may affect selection. Label stratification for the two training
sources is intentional and limited to satisfying the published class quotas.

| Dataset | Train | Development | Calibration | Options |
| --- | ---: | ---: | ---: | ---: |
| DBpedia-14 | 252 (18 per class) | 56 (4 per class) | 56 (4 per class) | 14 |
| UCI SMS | 252 (126 ham, 126 spam) | 60 (30 per label) | 60 (30 per label) | 2 |
| SNLI | 0 | 128 new groups | 0 | 3 |
| **Total** | **504** | **244** | **116** | — |

There are 864 examples. Train, development, and calibration groups and exact semantic
requests must be disjoint. The manifest uses the true source IDs and an explicit
source-group partition policy; do not alias source IDs to bypass split checks. SNLI is
development-only and is excluded from optimization, calibration, and checkpoint selection.

## Model and fixed training schedule

Use the rehearsal's pinned runtime and model recipe:

| Setting | Value |
| --- | --- |
| Base | `Qwen/Qwen3.5-0.8B-Base@dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, anonymous Hub access |
| Runtime | Python 3.12; Torch 2.14.1+cu130; torchvision 0.29.1+cu130; Transformers 5.18.0; PEFT 0.21.0 |
| Backbone and adapter | Frozen BF16 base; FP32 LoRA; rank 8, alpha 16, dropout 0, no bias |
| Targets and inventory | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `out_proj`; exactly 60 modules, 120 tensors, and 2,015,232 trainable parameters |
| Optimizer | AdamW over adapter parameters; learning rate `5e-4`; zero weight decay; global gradient clip `1.0`; seed `20261005` |

Give the remote trainer only the 504 train records and their labels. Reuse the rehearsal's
candidate-only loss and deterministic `epoch_examples` schedule. Shuffle records and option
order from the frozen seed and remap the gold answer by stable option ID. Run exactly two
epochs: 1,008 single-example training forwards, consumed as four sequential unpadded
microbatches per update with each loss divided by four. This is exactly 252 optimizer
updates. Do not stop early or select a checkpoint from development, calibration, or SNLI
scores.

Persist and commit the update-126 snapshot as a save-only checkpoint. Do not score or reload
it. Persist and commit the update-252 snapshot before evaluation. The final update-252
adapter is selected in advance; the midpoint is retained only as recovery evidence. Require
finite nonzero adapter gradients, no base gradients, and at least one changed adapter
tensor. Record per-update losses and gradient evidence. A failed training run remains
failed even if some evidence was saved.

## Evaluation presentations and forward budget

Use the same semantic option IDs and order panel for the base and final adapter. Development
and SNLI inference requests contain no answer labels. Keep gold labels local. Give each
presentation an opaque stable ID and explicit `order_ids`; do not encode its gold answer in
the ID. Save logits aligned to semantic option IDs.

- DBpedia development: canonical order, its 13 other cyclic rotations, reverse order, and
  the reverse order's 13 other cyclic rotations. These are 28 unique orders; each label
  occupies each position twice.
- SMS development: both possible orders.
- Calibration: original order only for all 116 rows.
- SNLI development: all six orders.

| Evaluation pool | Presentations per evaluation state |
| --- | ---: |
| DBpedia development, 56 × 28 | 1,568 |
| SMS development, 60 × 2 | 120 |
| Calibration, 116 × 1 | 116 |
| SNLI development, 128 × 6 | 768 |
| **Total** | **2,572** |

The maximum is 6,184 model forwards: 2,572 base evaluations, 1,008 training forwards,
2,572 final-adapter evaluations, and 32 fresh-reload parity forwards. Choose the parity
subset before seeing outputs: the first 32 presentation IDs in sorted order. Require exact
adapter tensor keys, shapes, and values after save and fresh reload, identical winners, and
maximum absolute logit difference at most `1e-3`. This checks 32 presentations only; do not
claim full-evaluation reload parity.

Save base outputs before training. Persist midpoint and final adapter snapshots, progress,
loss and gradient series, base and final per-presentation semantic logits, request and prompt
hashes, and a strict JSON receipt. Store adapter artifacts only in the existing personal
Modal Volume `reflex-rehearsal-artifacts`, under a unique `runs/<run-id>/` path. Hash and
verify the paths and updates. Return JSON and artifact paths, never Torch objects or weights.

## Preregistered scoring and uncertainty

Compare the base and final adapter on original-order rows as the primary accuracy result for
each task. Also report equal-task macro original-order accuracy for DBpedia and SMS. Report
SNLI accuracy separately. Keep the development, calibration, and SNLI results distinct.

For each task and model, report raw `T=1` and fitted-temperature NLL, summed-class Brier
score, ten-bin equal-width ECE, and macro-F1. Report risk-coverage curves grouped by tied
confidence and AURC descriptively. Do not choose a fixed abstention threshold from
development or SNLI. Use the existing tie policy: choose the lexicographically smallest
semantic option ID as the stable winner and automatically abstain on a tie. Report tie counts.
Also report accuracy for each order, minimum and maximum order accuracy, the count of
records correct in every option order, and the source-group rate with any semantic prediction
flip across its option orders.

Sampling units are source groups. Repeated option orders are paired measurements, not new
independent examples. Report 95% Wilson intervals for original-order correctness and
any-order flip rates by task. For original-order paired accuracy, report the adapter-minus-
base delta and both discordant counts: base-correct/adapter-wrong and base-wrong/adapter-
correct. Run a deterministic source-group bootstrap with 2,000 replicates and seed
`20261005`, stratified by dataset. Resample groups with replacement within each dataset and
keep both models' predictions paired. Report percentile 95% intervals for the accuracy
deltas and macro accuracy delta. Include these uncertainty results even when the point
estimates meet the learning gate.

Fit one global temperature separately for each model using only the 116 original-order
calibration labels. Give each DBpedia calibration row weight `1/56` and each SMS row weight
`1/60`, so the two task means have equal weight. Use the fixed 82-candidate grid: 81
log-spaced temperatures from `0.05` to `10`, plus exact `1.0`. Report raw and fitted NLL,
the selected temperature, and whether the selected candidate is a grid boundary. Freeze
each model's temperature before applying it to development and SNLI. No development or SNLI
label may influence temperature selection. The grid result is a discrete search, not a
continuous global optimum.

The engineering learning gate is:

1. DBpedia/SMS equal-task macro original-order accuracy improves by at least 5 percentage
   points over the base.
2. Neither DBpedia nor SMS original-order accuracy drops by more than 5 points.
3. The equal-task macro any-order semantic flip rate decreases by at least 25% relative to
   the base. If the base flip rate is zero, the final rate must also be zero.
4. SNLI original-order accuracy drops by no more than 5 points.

These are engineering thresholds, not significance claims. Failure calls for a bounded
diagnosis or correction, not broad scaling. Preserve regressions and failed results. Passing
the point gate does not replace the preregistered uncertainty report or justify a broad
quality claim.

## Compute and launch boundary

Use one ephemeral Modal A10 worker with 2 physical CPU cores, 16 GiB memory, one maximum
container, zero minimum and buffer containers, zero retries, a 300-second startup timeout,
a 3,600-second function timeout, and single use. Do not create an endpoint, schedule, or
warm pool. These are resource and time limits, not a dollar ceiling. Check the personal
account's current billing state separately before any paid launch.

The runner must verify the personal Modal profile/workspace, reject credential environment
overrides, and measure uploaded module hashes against the reviewed source map before model
load. Upload only the approved Python modules, package files, and payloads. Recovery may
adopt progress only when strict JSON, the unique run nonce, data/protocol/model pins, and
both source maps match; recovered evidence from an exception remains a failed run. The
default command is plan-only. A paid run requires a separate explicit launch action after
review of the exact-source diff, passing checks, frozen protocol, and run plan.
