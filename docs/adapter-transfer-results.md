# Adapter transfer results

The selected SNLI-trained adapter scored higher than the fresh base on these
small, previously inspected development samples. The reading gain is clearer;
the original-order cause-and-effect gain remains uncertain.

## Original choice order

| Task | Questions | Fresh base | Adapter |
| --- | ---: | ---: | ---: |
| Reading (BoolQ) | 32 | 17/32 (53.1%) | 27/32 (84.4%) |
| Cause and effect (COPA) | 32 | 24/32 (75.0%) | 27/32 (84.4%) |

Percentage-point change is the adapter percentage minus the base percentage.
The ranges show uncertainty in that change.

| Task | Change | 95% range |
| --- | ---: | ---: |
| Reading (BoolQ) | +31.3 points | +9.4 to +53.1 |
| Cause and effect (COPA) | +9.4 points | −12.5 to +31.3 |

COPA's range includes both losses and gains, so this sample does not show a
clear original-order change.

## Reordered choices

Each question was scored once in its original choice order and once with the
choices reversed. These are two views of 32 questions per task, not 64
independent questions. “Answer changed” counts questions where the selected
answer differed between the two orders.

| Task | Fresh base correct | Adapter correct | Answer changed, base → adapter |
| --- | ---: | ---: | ---: |
| Reading (BoolQ) | 39/64 (60.9%) | 50/64 (78.1%) | 21/32 → 4/32 |
| Cause and effect (COPA) | 36/64 (56.3%) | 58/64 (90.6%) | 28/32 → 6/32 |

| Task | Change across both orders | 95% range |
| --- | ---: | ---: |
| Reading (BoolQ) | +17.2 points | +1.6 to +31.3 |
| Cause and effect (COPA) | +34.4 points | +26.6 to +42.2 |

## Run and limits

The run completed 272 model scoring passes: 128 for the base, 128 for the
adapter, and 16 saved-adapter reload checks. It performed zero training. The
reload checks matched the prompt, token counts, and selected answers; scores
were identical. Exact hashes and interval details are in the
[verification summary](verification/adapter-transfer-summary.json) and the
frozen [protocol](adapter-transfer-protocol.md).

The Modal app stopped with no remaining tasks. A delayed, workspace-wide
billing snapshot showed USD 2.32 metered and USD 0 billed after credits. This
is not a per-run bill.

Both datasets are previously inspected development data, not sealed tests;
each task has only 32 independent questions. These questions were excluded
from adapter training, but pretraining overlap is unknown. The comparison
measures the entire selected adapter against a fresh base, so it cannot isolate
the contribution of SNLI training. It sets no new quality gate and does not
establish broad superiority.

COPA is BSD-2-Clause; retain its [license notice](licenses/copa.txt). BoolQ is
CC BY-SA 3.0; retain its [source notice](../data/notices/BOOLQ-DATA-NOTICE.txt).
See the [COPA](baseline-protocol.md) and
[broader-data](broader-baseline-protocol.md) protocols for dataset details.
