# Two-task development comparison, version 1

Frozen before any model scores on these samples, 2026-10-04. The completed
[COPA protocol](baseline-protocol.md) and results remain a separate experiment.

## Question and sample

Do the same three pinned models make accurate, stable decisions on yes/no
reading comprehension and three-way sentence inference?

- **BoolQ development:** 32 distinct source groups; native yes/no decisions.
- **SNLI 1.0 development:** 32 distinct premise groups; an explicit recast into
  three runtime labels, entailment, neutral and contradiction.

Use only the official labeled development files. The SNLI archive may contain
other splits, but open only its development member and attribution/license
material. Do not inspect train or test examples for this comparison.

Pin the exact downloaded bytes and the extracted development bytes by SHA-256
in the committed preparation recipe before launching inference. Preserve the
original licenses and attributions. Do not commit downloaded corpus text.

Group BoolQ rows by connected components sharing a whitespace-normalized,
case-folded passage or nonblank title. Group SNLI by shared caption/premise
source ID or normalized premise. Deduplicate exact normalized input pairs.
Exclude SNLI rows with no consensus label (`-`). Do not balance by gold label,
model output or prompt length. Select groups by SHA-256 of
`reflex-broader-v1:{dataset}:{group_id}`, then choose a representative by a
stable input/ID hash within each group. Record the exact implementation and
resulting file hashes. Each dataset contributes 32 independent source groups;
this grouping does not establish statistical independence of all semantics.

## Models and presentations

Reuse the exact Qwen, Intern and Kev checkpoints, prompt encoders, raw-logit
scoring, published temperatures and inference runtime from the COPA experiment.
See its protocol and the committed connector constants for immutable revisions.
There is no prompt tuning, fine-tuning, temperature fitting or model selection
using the new scores.

Original semantic option order is `yes, no` for BoolQ, and
`entailment, neutral, contradiction` for SNLI. Every model sees all permutations
in deterministic `itertools.permutations` order, keeping semantic IDs attached
to their text. Index zero is the original order. This yields 64 BoolQ and 192
SNLI presentations per model: **768 scored presentations total**, plus the
three existing connector verification forwards.

Model workers receive IDs and requests only. Gold answers remain in the local
analysis process. Each request must fit the existing 2,048-token cap under all
model encoders. Overflow or a malformed/incomplete result fails the experiment;
do not silently remove or replace records after seeing model output. Preserve
failed attempts and document any prospective protocol amendment separately.

## Analysis fixed in advance

Report each dataset separately, with original-order accuracy as the primary
metric and Wilson 95% intervals for its 32 groups. Also report macro-F1, NLL,
summed-class Brier score and ten-bin top-label ECE for raw and shipped-temperature
probabilities; gold and predicted position counts; accuracy by permutation;
and the number of source groups whose semantic decision changes in any order.
Use the existing deterministic semantic-ID tie policy and disclose ties.
Include paired original-order correctness counts for each model pair.

Permutations are repeated measurements, never extra independent questions.
Do not pool the two datasets into an unqualified headline accuracy. These are
small development measurements, not latency/cost benchmarks or broad rankings.
BoolQ is a disclosed Kev supervised source, while selected-row overlap is
unknown. Kev's MultiNLI exposure is related to SNLI; SNLI row overlap is unknown.
Intern's training overlap and all base-pretraining overlap remain unknown.
Do not claim unseen-family, contamination-free, or Reflex improvement.

## Execution and next step

Use the personal Modal profile and three ephemeral A10 containers at most,
each with the existing 300-second startup and 900-second function limits,
zero application retries and no warm pool, endpoint, schedule or volume.
Preserve source fingerprints, input/protocol hashes, raw scores, model loading
evidence and teardown status. Time limits are not dollar caps; actual billed
cost is a separate measurement.

After this comparison, proceed to a separate small LoRA mechanics rehearsal.
COPA, BoolQ and SNLI are reserved for development and must not enter that
training set. The rehearsal checks gradient flow, learning on its own training
examples, and adapter save/reload. Fitting those examples establishes working
training machinery, not generalization or competitive quality.
