# Controlled-mixture training protocol: initialization seed 2

**Status:** Frozen candidate for a stability repeat; no run was launched by this document.

## Incorporated protocol

This protocol incorporates the complete [original controlled-mixture protocol](mixture-training-protocol.md)
by reference. Its exact bytes are part of this candidate's contract and must have SHA-256
`06d3014a45ebeb19500ba2ab002f0595c42b44e561fbb9efc329af0d02a1030c`. Any change to those
bytes invalidates this candidate. Every requirement in that protocol applies unless the
single initialization-seed change below explicitly overrides it.

## Sole recipe change and inherited invariants

| Area | Seed-2 requirement |
| --- | --- |
| Initialization seed | Change only from `20261005` to `20261006`. |
| Data and splits | Keep every input, manifest, recipe, label, split, source group, and data hash identical. |
| Schedules | Keep schedule seeds, example and option orders, interleaving, repeats, presentations, and schedule hashes identical. |
| Training recipe | Keep model and revision, runtime pins, BF16 base, FP32 LoRA rank 8 / alpha 16 / dropout 0, AdamW, LR `1e-4`, zero weight decay, gradient clipping at 1.0, 252 updates, four microbatches per update, and final update-252 checkpoint identical. |
| Evaluation and decisions | Keep all panels, calibration rows and fitting rules, seven gates, paired bootstrap seed `20261007`, 2,000 draws, and interval method identical. |
| Execution limits | Keep workers, A10 profile, timeouts, retry policy, forward/token limits, and artifact handling identical. |

This is an initialization-variation check, not a data-order experiment or configurable seed
sweep. Use run ID `mixture-2026-10-04-r2` and fresh initialization, `real_only`, and
`synthetic_mix` directories. After initialization, verify its tensor digest differs from
R1's `330beae6f642222a6364a7d8ea66d59e955b8e02b79c2678abd13dddbc16f85c`. Each arm must
independently verify equality to this run's shared initialization before optimizer creation.

## Interpretation and reproducibility

Keep R1 and this repeat as separate paired results; do not select a winning seed. R2 must
pass the same seven advancement components to count as a successful repeat. Two seeds do
not establish broad superiority or training-seed confidence intervals. Data-order schedules
remain fixed, so this protocol varies initialization only.

R1 remains reproducible at source commit
`2e46d664dc970cd3d6bced19b4f5524fc68d01ae`. The current analyzer verifies current source
bytes, so reproduce R1 from that source checkout; do not weaken its source validation to
accommodate this repeat. The original R2 real-only pilot remains failed, as recorded in the
original protocol and results.
