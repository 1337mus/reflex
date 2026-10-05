# Runtime-rule study readiness

The CPU-side input preparation, saved-result reuse, analysis rules, and worker settings are verified. The independent review of the six cache, statistics, and contracts files approved them with no findings. No new study model was trained or scored, so there is no performance result yet.

The input and schedule slice was delivered in [c883de9](https://github.com/1337mus/reflex/commit/c883de9f0053e82554fab332fbb5b86a980a54ea), with [passing GitHub checks](https://github.com/1337mus/reflex/actions/runs/37296034094). The reviewed cache, statistics, and worker-contract files are signed and pushed in [074e483](https://github.com/1337mus/reflex/commit/074e4836c75c59a40373ef0dba06a4bd9307d8d6); [its exact-commit CI passed](https://github.com/1337mus/reflex/actions/runs/37300810702).

## Data and prompt preparation

The loader checks pinned file hashes. The prompt compiler records token counts and choice order. It also hashes the prompt text and token IDs so a worker can verify it received the same request.

| Check | Result |
| --- | ---: |
| Files verified by the data loader | 87 |
| Planned model inputs checked | 10,808 |
| Total planned input tokens | 2,544,210 |
| Longest prompt / limit | 695 / 2,048 tokens |
| New calibration or test files opened | 0 |
| Clean-checkout full suite | 785 passed |

The schedules, prompt identities, and token totals matched the earlier preparation. The integrated work remained CPU-only and did not load model weights.

A token is a piece of text counted by the pinned tokenizer; it can be shorter or longer than a word.

## Training plans

| Measure | Continued practice | New practice (runtime_mix) |
| --- | ---: | ---: |
| Times examples are shown | 1,344 | 1,344 |
| Learning steps | 336 | 336 |
| Different examples | 1,008 | 1,120 |
| Training tokens | 225,924 | 349,296 |
| Longest training prompt | 626 tokens | 686 tokens |

New practice uses 123,372 more training tokens, about 55% more, despite the same number of learning steps. The comparison measures the full replacement plan; it does not isolate the effect of text volume.

## Reused results and worker budgets

The study reuses the selected adapter's saved results for seven earlier tasks. The cache contained all 3,782 expected rows, and an independent join matched them to the old receipt exactly. Those saved rows account for 828,622 cached input tokens. All 60 historical source pins still match.

| Worker | Model passes | Input tokens |
| --- | ---: | ---: |
| Unchanged adapter, new development only | 164 | 92,958 |
| Continued practice | 5,322 | 1,163,940 |
| New practice | 5,322 | 1,287,312 |

Each model pass is one request used in training or scoring. The fixed worker settings specify one A10 per worker, a 900-second limit for the unchanged worker and 3,600 seconds for each training arm, 2 CPUs, 16 GiB of memory, and no retries. These are frozen settings; no worker has launched. No new calibration/test records were read, Torch was not imported, and no new study outputs were created.

## Fixed pass/fail checks

There are 21 pass/fail checks, all fixed in advance. Against continued practice, new practice must gain at least 5 percentage points on the average of the two new question families. The bottom of its uncertainty range must be above zero, neither family may lose accuracy, and the score on each of seven earlier tasks may fall by no more than 5 points. Against the unchanged adapter, the same gain and family rules apply, along with the seven earlier-task limits. Its uncertainty range is descriptive and does not need to be above zero.

Point comparisons use exact fractions. An exact zero stays zero, so rounding cannot make it pass a check that requires a positive value. An independent 3,946-row toy check matched all 20 reported uncertainty ranges and all 21 pass/fail results, including checks designed to fail.

The validation checkpoint includes 68 focused tests, 36 additional rejection probes, the clean 785-test suite, Ruff lint and format checks, `mypy src`, strict mypy for the three study modules, and both CLI fixture checks. All passed. The independent final review approved the frozen six-file slice with no findings.

## Limits and remaining work

The new development panel has 28 groups: 14 routing and 14 tool-choice groups. It is a small, previously inspected development panel, not a sealed test. Each family has only two questions with more than one missing input. In these questions, all possible values lead to the same answer. The panel does not cover cases where those values lead to different answers. This is a one-seed engineering study, not evidence of broad capability. Token exposure also differs between the two training plans.

The authenticated worker bundle, payload contract, result contract, and GPU runner still need implementation and review. Exact-commit CI passed for 074e483. The schedules are fixed. Before launch, freeze the concrete worker payloads and pass the launch checks. No run has started and no performance gain has been measured. New calibration and test files remain sealed and unused.

Counts, hashes, and check evidence are in [runtime-rule-study-readiness.json](verification/runtime-rule-study-readiness.json). The accepted design is [runtime-rule-study-protocol.md](runtime-rule-study-protocol.md).
