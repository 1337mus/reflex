# Runtime-rule study readiness

The CPU checks passed. The GPU runner is still being built. No new training result is available yet.

The first schedule and evaluation-panel slice is in [c883de9](https://github.com/1337mus/reflex/commit/c883de9f0053e82554fab332fbb5b86a980a54ea). Its [GitHub checks passed](https://github.com/1337mus/reflex/actions/runs/37296034094).

## Data and prompt preparation

The data loader now checks the exact file hashes before use. The prompt compiler records each input's token count and answer-choice order. It also hashes the exact text and token IDs. A token ID is the number the tokenizer assigns to a piece of text. These hashes let the GPU runner check that it received the same input.

| Check | Result |
| --- | ---: |
| Files verified by the data loader | 87 |
| Tests passed in a clean checkout | 717 |
| Planned model inputs checked | 10,808 |
| Total input tokens | 2,544,210 |
| Longest prompt / limit | 695 / 2,048 tokens |
| New calibration or test files opened | 0 |

The integrated check reproduced the earlier schedules, prompts, and token totals exactly. The clean checkout contained no private processed data or cached tokenizer. This confirms the tests can run from repository files alone. A separate local check used the four approved new training/development files and the pinned tokenizer. It did not load model weights or run the model.

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

Root's local checks passed, including the full test suite (**717 passed**), code style, type checks, and fixture commands. No model was trained or scored, so no performance gain has been measured. Before a run:

- Check that saved results use the same questions, tokenizer, and model.
- Finish the result calculator and the data package sent to each GPU worker.
- Finish the GPU runner and launch checks.

Machine-readable counts, hashes, and check evidence are in [runtime-rule-study-readiness.json](verification/runtime-rule-study-readiness.json). The accepted protocol is [runtime-rule-study-protocol.md](runtime-rule-study-protocol.md).
