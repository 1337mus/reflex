# Targeted study R2: training finished, cloud controller stopped

**Date:** October 5, 2026. **Run:** `targeted-reasoning-2026-10-05-r2`.

Both arms finished: control used old practice; treatment added HANS/WinoGrande practice. There is no new accuracy result because the whole-run completion record failed required checks.

| Version | Training steps | Test answers saved | Saved-adapter checks |
| --- | ---: | ---: | ---: |
| Unchanged reference | 0 | 6,182 | — |
| Control: old practice | 1,200 | 6,182 | 32 of 32 |
| Treatment: added HANS/WinoGrande | 1,200 | 6,182 | 32 of 32 |

The test counts include repeated questions with different answer orders. Each saved adapter passed all 32 reload checks: its learned weights and chosen answers matched exactly, with zero score difference. All 120 LoRA weight arrays changed; the base model stayed frozen. These checks confirm training and saving worked. They do not show whether reasoning improved.

The cloud controller stopped before completing the run record, so the accuracy tool did not open reserved answers or scores. All 24 learning checks remain unevaluated. The interruption's cause is unknown. Keep the previously selected `snli_mix` update 378; R2 shows neither improvement nor a learning failure.

Both adapters and their evidence remain unchanged, and reserved content stays closed. Names-and-sizes checks found all three saved files per trained arm on Modal; no weights were downloaded or new compute used. At 23:32:45 UTC the app was stopped with zero tasks. Workspace usage was about $4.86 metered and $0 billed; these delayed, revisable totals are not per-run cost.

The system log says a worker disappeared and inputs might be rescheduled; available records cannot confirm another attempt. The controller asked Modal to stop the app after the run ended. A later “user stopped from CLI” label does not show Raj initiated it. No automatic retry is authorized. Future explicit review could consider saved work without training, but current rules do not allow marking this run successful. See the [training protocol](targeted-training-protocol.md), [execution protocol](targeted-execution-recovery.md), [verification](verification/targeted-training-r2-interruption.json), and unchanged [R1 report](targeted-training-interruption.md).
