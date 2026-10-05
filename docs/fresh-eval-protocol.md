# Fresh public-source evaluation protocol

**Status:** Fixed before model execution. Source review and exact-commit CI must pass before launch.
This evaluation does not train or select a checkpoint. It cannot turn an earlier failed
result into a pass.

## Question and limits

Compare the already selected `snli_mix` adapter at update 378 with the untouched
`Qwen/Qwen3.5-0.8B-Base` revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. The adapter selection is recorded in
`data/evaluations/adapter-transfer-v1-selection.json`. Keep the existing BF16 base and
FP32 LoRA runtime, scorer, and prompt rendering. Do not train, tune, choose another
checkpoint, or change the closed calibration and test pools. Preserve earlier failures as
failures.

These are public benchmark development results, not sealed tests or known-unseen data.
Public HF previews were exposed during metadata research. Exclude the one identified ARC
card example by its exact question hash before selection. Other viewer preview identities
were not retained, so their overlap with this panel remains unknown.
Qwen pretraining overlap is unknown. Do not claim an uncontaminated or generally
representative reasoning score. Other models are outside this run.

## Sources and fixed panels

Use only these labeled splits. Source pages are public and accessible without an account.
The exact download revisions, file hashes, licenses, parser versions, and split counts must
be pinned in the run manifest before launch; reject changed or missing pins.

| Dataset | Fixed sample | Task and gold format | License evidence |
| --- | ---: | --- | --- |
| [HANS](https://github.com/tommccoy1/hans) evaluation set | 300; 10 from each of 30 subcases | Classify a premise/hypothesis pair as entailment or non-entailment | Repository says MIT; it does not itemize separate data terms |
| [WinoGrande](https://github.com/allenai/winogrande) `dev` | 200 of 1,267 | Choose which of two options fills a sentence blank using commonsense | Dataset says CC-BY with no version; codebase is separately Apache-2.0 |
| [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc) `validation` | 200 of 299 | Choose the correct answer to a grade-school science question | Dataset card says CC-BY-SA-4.0 |

The WinoGrande official README withholds test labels; use its labeled development split.
Use ARC validation for this comparison. Open only the approved members of downloaded
source archives. Keep the project's existing calibration and test records closed.
Keep source files and outputs private and do not redistribute dataset content.

Select each panel using seed `20261011`: rank by the SHA-256 of the UTF-8 string
`fresh-eval-v1:20261011:{dataset_id}:{raw_source_id}`, then break ties by raw source ID.
Take ten questions per HANS subcase and 200 per other task, independently of answers.
Freeze deterministic parsers and source revisions
with file hashes. Deduplicate semantic-exact requests within and across panels. Compare
canonical requests and stable IDs against the actual records used to train `snli_mix`
before scoring. Record the known ARC preview exclusion and the limits of this check.
If pins, parsing, deduplication, or
training-overlap checks fail, stop before model execution.

## Presentation and scoring

For each of the 700 questions, score the original option order and a left cyclic rotation
by one position. Preserve semantic option IDs through rotation. Sort all work by
`dataset_id`, `record_id`, and `order_index`. The selected winner is the semantic option
ID with the highest candidate score; use the existing scorer's fixed tie rule. Pin the
compiled prompt hash, rendered token IDs and counts, candidate IDs, source IDs, and parser
hashes. Reject prompts above 2,048 tokens; never truncate.

Keep gold labels on CPU. Send only label-free requests and identities to the GPU. Each
state receives 1,400 presentations. Before scoring, verify exact equality between the
loaded adapter tensors and its existing saved safetensors. Fresh-load it and rerun both orders
for the first four sorted record IDs in each task (24 checks); require identical semantic
winners and maximum candidate-score difference at most `0.001`.

| Work | Maximum presentations |
| --- | ---: |
| Base and adapter, 700 questions × two orders × two states | 2,800 |
| Fresh adapter reload, 12 questions × two orders | 24 |
| **Total** | **2,824** |

At 2,048 tokens per presentation, the maximum input budget is **5,783,552 tokens**.
Record actual totals, prompt hashes, tensor hashes, and reload evidence. Keep question text
out of progress and result reports; report stable IDs only where needed for audit.

## Metrics fixed before launch

Report each task separately and each state separately: original-order correct/count and
accuracy, mean accuracy across the two orders, and semantic winner changes between orders
with the task's question count as denominator. Report paired adapter-minus-base differences
for original-order accuracy and two-order mean accuracy, each with a 95% paired bootstrap
interval.

Use 2,000 bootstrap resamples, seed `20261012`, sorted task and group IDs, and type-7
percentile interpolation. For HANS, resample its 30 subcases as clusters; within each
subcase, average over questions and orders. For WinoGrande and ARC, resample the 200
question IDs within each task. Keep each question's two orders and both model states paired
in every draw. Report HANS's three heuristic and 30 subcase results as descriptive
diagnostics. Do not resample based on observed scores, tune on these results, combine tasks
into an overall reasoning score, or create a new learning threshold or promotion rule.

## Execution and stop rules

Run only after root freezes this document, the source pins, parser and scorer identities,
and the exact execution source. Require signed exact-source CI and root review. Before
launch, check fresh personal Modal usage and available budget, read the current A10 rate,
calculate and record a bounded cost estimate for the 300-second startup and 1,800-second
function limits, and reserve against the roughly `$300` budget. Do not launch if usage,
rate, or budget cannot be verified.

Use the personal Modal profile `reflex-personal`, workspace `rajath-61258`, environment
`main`, SDK `1.6.1`. The laptop remains CPU-only. Run one single-use A10 worker with two
CPU cores and 16 GiB memory, a 1,800-second function timeout, and a 300-second startup
timeout. Set retries, minimum containers, and buffer containers to zero; maximum
containers to one; scale-down window to two seconds. Use one attempt marker and exclusive
output path. Do not automatically rerun the same data.

On any failed, partial, or unknown run, preserve its evidence, stop it, and verify that no
remote task remains active. Do not turn partial evidence into a pass. Complete the final
source review, sign and deliver the source commit, and require its exact-commit CI to pass
before launch. Independently recount the results, then deliver the report in a signed
descriptive commit on private `1337mus/reflex` main and verify its CI. Do not create a PR or Linear
ticket, contact employer services, or publish data or models. Deliver the report and
evidence at the exact committed revision.
