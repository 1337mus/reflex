# Targeted study: interrupted before training

**Date:** October 5, 2026. **Run:** `targeted-reasoning-2026-10-05-r1`.

The unchanged adapter completed evaluation. The local launcher became
unavailable before it recorded that result. The two training runs did not start. This attempt provides no
training result and no accuracy comparison.

| Part | Verified outcome |
| --- | --- |
| Source checks | Exact signed commit passed CI on attempt 2 |
| Unchanged adapter | All 6,182 evaluation calls completed; execution validated after recovery |
| Control training | Did not start |
| New-data training | Did not start |
| New trained adapters | None |
| GPU workers | App stopped; zero running tasks |

The launch log records a timeout while Modal's connection heartbeat ran.
The original local process is gone, and it did not write a final host receipt.
The exact reason the process exited is not established. The remote reference
result was recovered and checked without starting another job.

The new test answers, predictions, and accuracy remain unopened. We did not
calculate any of the 24 learning checks. Earlier model results remain unchanged.

At 1:29 p.m. Pacific, Modal reported about **$3.05 metered** and **$0 billed**
for the workspace. Billing can arrive late or change. These are workspace totals,
not the cost of this attempt.

The automatic continuation has stopped. No retry was launched. Before another
attempt, the launch process needs a reviewed way to survive a lost laptop
connection and recover its run state. The fixed data and failed attempt remain
preserved.

See the [fixed protocol](targeted-training-protocol.md) and
[verification record](verification/targeted-training-interruption.json).
