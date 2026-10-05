# Controlled-mixture training results (R1)

With reasoning practice, balanced-SNLI accuracy rose from 53.6% with real data only to 67.8%, and all seven predeclared R1 checks passed. This is one training seed; it supports a repeat, not a claim of broad superiority or release.

## Balanced SNLI

| Model or training recipe | Correct / 1,152 | Accuracy |
| --- | ---: | ---: |
| Qwen3.5-0.8B-Base | 447 | 38.80% |
| Real data only | 617 | 53.56% |
| With reasoning practice | 781 | 67.80% |
| Intern-Decision-0.8B | 670 | 58.16% |
| Kev-0.8B | 877 | 76.13% |

Across 64 source groups, the measured gain was 14.24 percentage points; the 95% interval (+10.76 to +17.71 pp) describes uncertainty from sampling those groups. The 192-record panel scores each record in six choice orders; the interval resamples groups, not orders as independent examples. Reference scores come from the earlier [balanced SNLI diagnostic](snli-diagnostic-results.md).

The labels mean the premise supports, rules out, or does not settle the claim. In original order, the model got 61/64, 45/64, and 24/64 cases right, respectively; on 24/64 undecided cases it wrongly chose “supports.” This describes the error pattern, not its cause.

## Paired training and retention

Both arms started from the same weights and got 1,008 presentations: 504 shared real examples plus 504 arm-specific ones. The reasoning-practice arm replaced its 504 real examples with 500 synthetic records and four repeats, keeping presentation counts equal while token totals differed because examples had different lengths. Its learning rate was also lower than in R2, so the full rise from R2’s 36% to 67.8% cannot be credited to synthetic practice alone.

| Original-order task | Qwen base | Real data only | With reasoning practice | Saved R2 final adapter |
| --- | ---: | ---: | ---: | ---: |
| DBpedia-14 | 31/56 | 54/56 | 56/56 | 50/56 |
| SMS Spam | 26/60 | 58/60 | 58/60 | 58/60 |

All seven predeclared checks passed:

| Check | Observed difference | Status |
| --- | ---: | --- |
| Balanced SNLI, practice vs. real data only | +14.24 pp (95% interval +10.76 to +17.71) | Pass |
| Balanced SNLI, practice vs. base | +28.99 pp | Pass |
| DBpedia selected order, practice vs. real data only | +1.98 pp | Pass |
| SMS selected order, practice vs. real data only | −0.83 pp | Pass |
| DBpedia original order vs. saved R2 adapter | +10.71 pp | Pass |
| SMS original order vs. saved R2 adapter | 0.00 pp | Pass |
| Original SNLI panel vs. base anchor | +28.91 pp (94/128 vs. 57/128) | Pass |

These checks do not reverse the earlier [R2 SNLI non-regression failure](real-pilot-results.md): its final adapter scored 1/128 against the 57/128 base anchor, below the preregistered limit.

## Execution and verification

Each arm completed 252 updates. Input tokens (text units processed by the model): 260,662 for shared initialization; 180,702 vs. 199,959 for training; 876,540 per arm for final evaluation; 7,371 per arm for reload checks. Both adapters reloaded exactly and picked the same winners across 32 checks (maximum logit difference 0.0). Storage checks passed for all 15 remote files; both apps stopped with no tasks remaining. A delayed workspace-wide billing snapshot showed $1.31 metered and $0 after credits, not a per-run invoice; the exact value is in the [verification summary](verification/mixture-training-summary.json).

Local evidence paths: `artifacts/mixture-2026-10-04-r1-receipt.json`, `artifacts/mixture-2026-10-04-r1-analysis.json`, `artifacts/mixture-2026-10-04-r1-storage-verification.json`, and `artifacts/mixture-2026-10-04-r1-teardown-billing.json`. These private files are not GitHub links; the tracked [verification summary](verification/mixture-training-summary.json) provides hashes and independent checks. See the frozen [protocol](mixture-training-protocol.md).

R1 reproduction requires source commit `2e46d664dc970cd3d6bced19b4f5524fc68d01ae` because the analyzer checks exact source bytes. Next, repeat the same data, schedule, and checks with initialization seed `20261006` to test repeatability.
