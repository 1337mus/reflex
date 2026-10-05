# COPA baseline pilot results

The frozen 32-question COPA development pilot completed on 2026-10-04. On this small slice, Kev had the highest original-order accuracy and no forced-choice flips; the results show no demonstrated advantage for a Reflex-trained method because no Reflex adapter has been trained or evaluated. Treat this as a reproducible development comparison, not a broad benchmark or evidence of general superiority.

## Primary results

| Model | Original order correct / 32 (95% Wilson interval) | Forced-choice flips after reversal / 32 | Reversed-order correct / 32 | Peak allocated GPU memory |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B-Base | 22/32 (51.4–82.0%) | 29 | 13/32 | 3,038,318,592 bytes |
| Intern-Decision-0.8B | 26/32 (64.7–91.1%) | 8 | 30/32 | 3,456,651,264 bytes |
| Kev-0.8B | 31/32 (84.3–99.4%) | 0 | 31/32 | 3,113,830,400 bytes |

Intervals are Wilson 95% intervals for the 32 original-order questions. Reversed presentations reuse those same questions and are a separate paired order-sensitivity check, not additional independent examples. The answer key contains 11 `choice1` and 21 `choice2` labels. Qwen selected the second presented alternative on 31/32 original-order items and 30/32 reversed-order items; its reversed-order predictions mapped back to only 13/32 correct semantic answers. A fixed `choice2` majority reference scores 21/32 on the original order.

Kev's zero observed flips describe these 32 paired cases only; they do not prove option-order invariance.

The analysis reports these paired correctness counts:

| Pair (left vs right) | Both correct | Left only correct | Right only correct | Both wrong |
|---|---:|---:|---:|---:|
| Qwen vs Intern | 16 | 6 | 10 | 0 |
| Qwen vs Kev | 21 | 1 | 10 | 0 |
| Intern vs Kev | 26 | 0 | 5 | 1 |

## Probability metrics

Each pair below uses the same 32 original-order logits: raw scores at `T=1` and the checkpoint or published shipped temperature. NLL penalizes low probability on the correct label, Brier is squared probability error, and ECE summarizes confidence-versus-accuracy gaps across bins. Lower is better for all three; ECE is especially noisy at this sample size.

| Model | Temperature | NLL | Brier | ECE |
|---|---|---:|---:|---:|
| Qwen raw and shipped | `1.0` | 0.5808 | 0.3934 | 0.0871 |
| Intern raw | `1.0` | 0.3804 | 0.2531 | 0.0654 |
| Intern shipped | `2.747760550703` | 0.4753 | 0.3021 | 0.1428 |
| Kev raw | `1.0` | 0.1615 | 0.0814 | 0.0839 |
| Kev shipped | `2.3510958125672174` | 0.3125 | 0.1659 | 0.2182 |

The shipped temperatures worsened all three probability metrics for Intern and Kev on this pilot. Do not refit them on these 32 examples or generalize this calibration result. Intern's temperature is the Hub default; its published preset is XTuner-bound and this runtime did not validate that preset. Kev's exact value was loaded from pinned checkpoint metadata. Temperature scaling changed confidence metrics, not top-1 accuracy.

## Method and limits

The frozen [protocol](baseline-protocol.md) used the licensed COPA development split, a deterministic 32-item sample, two semantic option orders, FP32, eager attention, and no inference cache. It kept the answer key local; remote requests contained unlabeled prompts. The exact model revisions were Qwen `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, Intern `85a0cc5a99d67ea8d56dfe98115689212867171d`, and Kev `9a45d25eb2ab761841196625383fa1dff0e56c1e`. Maximum observed input lengths were 69, 186, and 43 tokens, respectively. All three loaded with clean public loading diagnostics and FP32 parameters.

Kev's upstream project declares PyTorch `>=2.6,<2.9`; this run used `2.14.1+cu130`, outside that range. The pinned reference path loaded and its parity probe passed in this environment. This is narrow compatibility evidence for that path and probe, not certification of the declared supported range or broader compatibility.

The run scored 192 presentations (64 per model) plus three recorded parity/probe forwards: one Intern and two Kev. Peak GPU memory values above are PyTorch allocator `max_memory_allocated` readings, not total device VRAM. This run did not measure latency, throughput, total device memory, or billed cost. COPA is one binary commonsense-causality task family on a development split; there is no sealed-test result here. Pretraining overlap cannot be ruled out, so “unseen” applies only to future Reflex post-training data unless stronger evidence becomes available.

## Reproduction and delivery record

- Frozen code: [`b9751fe9bbffcfb827334589269b8f903663a412`](https://github.com/1337mus/reflex/commit/b9751fe9bbffcfb827334589269b8f903663a412); [CI passed 139 tests plus lint, format, and mypy](https://github.com/1337mus/reflex/actions/runs/37249587550).
- [Canonical run receipt](verification/copa-baselines.json), SHA-256 `e988949ef8d1a54f36661a11d22e4b72337a9e214653da77efc2545e7b424549`.
- [Canonical analysis](verification/copa-baselines-analysis.json), SHA-256 `2135f6f70ba9bb8e2407bc4a5803593e971f1fb80e457964884522a574a6bc26`.
- [App teardown record](verification/copa-baseline-teardown.json): all comparison apps stopped with zero tasks; attempt 3 app `ap-ghprXHjOxD1ol0QlqbuWWG` ran 17:59:12–17:59:56 PDT.
- The protocol SHA-256 is `bee16b5568551a087aeef4e736d1b16916043f391ed0b519efb319cd1ab284ef`; records SHA-256 is `a8daefa5a7300cc6243f3af1208bbd2887d41c306416bb37bf95263748657c5b`; manifest SHA-256 is `4e60e82ea1622d4e069b4244aa596db64cafa03bdb9ec85d3c4d06d2c40e7635`.

Two earlier attempts are preserved for audit. Attempt 1 at `e0c946d` failed during CUDA runtime setup before model loading because peak-memory reset ran before explicit CUDA initialization; `a9ff2ea` added initialization first. Attempt 2 at `a9ff2ea` scored Qwen and Intern but Modal could not deserialize Kev's `torch.torch_version.TorchVersion` metadata on the local runner; `b9751fe` casts the version to a plain string and normalizes remote receipts through strict JSON before transport. See [attempt 1](verification/copa-baseline-attempt1.json) and [attempt 2](verification/copa-baseline-attempt2.json).

The next evidence step is a preregistered development expansion to two additional task types with 2–3 options, with source, license, and group/split pins recorded before scoring. Do that before competitive training. A small LoRA mechanics rehearsal is a separate future goal; no training was run or claimed by this result.
