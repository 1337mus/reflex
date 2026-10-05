# Runtime-rule study data readiness

The CPU checks passed. The GPU runner is still being built. No new training result is available yet.

**Reviewed source commit:** [c883de9](https://github.com/1337mus/reflex/commit/c883de9f0053e82554fab332fbb5b86a980a54ea). Its [GitHub checks passed](https://github.com/1337mus/reflex/actions/runs/37296034094).

## Training plans

| Measure | Continued practice | New practice |
| --- | ---: | ---: |
| Times examples are shown | 1,344 | 1,344 |
| Learning steps | 336 | 336 |
| Different examples | 1,008 | 1,120 |
| Text pieces | 225,924 | 349,296 |
| Longest training prompt | 626 tokens | 686 tokens |

Here, a token is one piece of text counted by the pinned tokenizer, not one word. New practice uses 123,372 more text pieces—about 55% more—even though both plans have the same number of learning steps. The comparison tests the whole replacement plan; it does not separate the effect of text volume.

## Evaluation plan

The new panel has 28 questions: 14 routing and 14 tool-choice questions. Moving the answer choices gives 164 presentations of those questions. Each trained arm also has 3,782 presentations of earlier questions to check retained skills. That gives 3,946 presentations per arm. The unchanged adapter will receive the same 164 new presentations. Reload checks will cover 32 presentations per arm.

The CPU check covered all 10,808 planned model inputs, including repeats. The longest prompt is 695 tokens, below the 2,048-token limit. Checks covered training, evaluation, the unchanged adapter, and saved-adapter reloads.

## Evidence and limits

Training and evaluation examples did not overlap. All 60 files used by earlier studies kept their original hashes. The new release's sealed calibration and test records remained unused.

Root's local checks passed, including the full test suite (**688 passed**), code style, type checks, and fixture commands. No model was trained or scored, so no performance gain has been measured. Before a run:

- Finish the data loader.
- Check that saved results use the same questions, tokenizer, and model.
- Finish the GPU runner and launch checks.

Machine-readable counts, hashes, and check evidence are in [runtime-rule-study-readiness.json](verification/runtime-rule-study-readiness.json). The accepted protocol is [runtime-rule-study-protocol.md](runtime-rule-study-protocol.md).
