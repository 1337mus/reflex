# Intern and Kev reference results

**Run:** `85b69940-11d3-4085-bfab-737707c51721` · **Status:** reference scoring passed; comparisons remain conditional development evidence.

Intern and Kev were compared with the saved final Qwen adapter on the same DBpedia-14, SMS, and SNLI development rows. The original Qwen learning gate remains failed because its SNLI non-regression component failed.
This reference comparison does not change that result. No overall winner is assigned.

## Original-order accuracy and semantic answer flips

Each value is a source-group count. A flip means the semantic top answer changed under at least one evaluated option order.

| Model | Dataset | Original-order correct / groups | Accuracy | Groups with a semantic flip |
| --- | --- | ---: | ---: | ---: |
| Qwen final adapter | DBpedia-14 | 50/56 | 89.3% | 11/56 (19.6%) |
| Qwen final adapter | SMS Spam | 58/60 | 96.7% | 0/60 (0.0%) |
| Qwen final adapter | SNLI | 1/128 | 0.8% | 99/128 (77.3%) |
| Intern Decision | DBpedia-14 | 51/56 | 91.1% | 19/56 (33.9%) |
| Intern Decision | SMS Spam | 52/60 | 86.7% | 13/60 (21.7%) |
| Intern Decision | SNLI | 8/128 | 6.2% | 121/128 (94.5%) |
| Kev | DBpedia-14 | 56/56 | 100.0% | 0/56 (0.0%) |
| Kev | SMS Spam | 45/60 | 75.0% | 8/60 (13.3%) |
| Kev | SNLI | 58/128 | 45.3% | 21/128 (16.4%) |

## Paired DBpedia/SMS comparison

Deltas are reference minus Qwen final, using the unweighted DBpedia/SMS task macro. The 2,000-replicate bootstrap resamples source groups independently within each dataset (seed `20261005`); option permutations are repeated measurements, not independent units.

| Reference | Reference task-macro accuracy | Qwen-final task-macro accuracy | Delta (95% paired group bootstrap interval) |
| --- | ---: | ---: | ---: |
| Intern Decision | 88.9% | 93.0% | -4.11 pp ([-12.03, +3.69] pp) |
| Kev | 87.5% | 93.0% | -5.48 pp ([-13.04, +1.97] pp) |

Both intervals include zero. These small paired development comparisons do not establish superiority or statistical significance.

## Temperature-fit diagnostic

Each model used its own temperature fit on the 116 original-order DBpedia/SMS calibration rows, with equal task weighting and the frozen 82-candidate grid. The calibration NLL values below are fit diagnostics, not held-out performance estimates.

| Model | Fitted temperature | Raw calibration NLL | Fitted calibration NLL |
| --- | ---: | ---: | ---: |
| Qwen final adapter | 1.672640 | 0.503486 | 0.375419 |
| Intern Decision | 0.862529 | 0.504036 | 0.498927 |
| Kev | 0.707107 | 0.305392 | 0.302938 |

## Scope and limits

- This is a small, selected development comparison on matched requests. It is not a test-set or contamination-free result, and no overall ranking is supported.
- The reference paths are precision- and encoding-mismatched: Intern and Kev use FP32, while the saved Qwen base is BF16 with an FP32 adapter. Each model also uses its own prompt and tokenizer.
- Kev discloses DBpedia-14 training exposure and related MultiNLI exposure; overlap with the selected rows is unknown. Intern, Qwen, and other pretraining overlap are not established.
- DBpedia redistribution terms require review. UCI grants CC BY 4.0 for the SMS collection, but upstream component licenses and message privacy remain unresolved. Raw corpus text and per-presentation outputs remain private.

The reference Modal app is stopped with 0 tasks remaining. The billing snapshot is workspace-wide, may lag recent usage, and does not attribute cost to this run.

## Reproduction evidence

| Artifact | SHA-256 |
| --- | --- |
| Reference receipt (`artifacts/real-pilot-references-2026-10-04-r1-receipt.json`) | `47fdf25955741b031594290a4b6205ea4fbda7b5a9bf793abc5c53e9c622a1ea` |
| Reference analysis (`artifacts/real-pilot-references-2026-10-04-r1-analysis.json`) | `f890c711144cf7aede1bbb8b76bf1e0bc2148de7a99fceab42a3420941b94115` |
| Modal teardown and billing snapshot (`artifacts/real-pilot-references-2026-10-04-r1-teardown-billing.json`) | `38e4aa87ab76aeee660698a5f2f414ac2c3a38703f30d0230ab4fdae0dcc5f21` |
| Original Qwen R2 analysis (gate status) (`artifacts/real-pilot-2026-10-04-r2-analysis.json`) | `f73689f01013ec7edc9f9b0d987a9d8bf587e003585ebdf9a0c5588ca9c338c1` |

The machine-readable aggregate and artifact pins are in [`verification/real-pilot-baselines-summary.json`](verification/real-pilot-baselines-summary.json).
