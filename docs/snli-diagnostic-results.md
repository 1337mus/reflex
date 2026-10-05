# Balanced SNLI diagnostic results (R1)

**Run date:** 2026-10-04. **Status:** complete; development-only evidence.

The balanced panel contains 192 SNLI development records from 64 source groups, with
three gold labels per group. Each checkpoint scored all six option permutations for
each record (1,152 presentations per model). The run completed all four model states:
4,608 scored forwards and three official auxiliary forwards, with no lifecycle failure.
The exact protocol, data-selection rules, model revisions, and interpretation limits are
in the [frozen protocol](snli-diagnostic-protocol.md) and [data note](snli-diagnostic-data.md).

## Results

| Model | All-order accuracy (correct / 1,152) | 95% source-group interval | Original-order accuracy (correct / 192) |
| --- | ---: | ---: | ---: |
| Qwen3.5-0.8B-Base | 38.80% (447) | 37.41–40.19% | 50.00% (96) |
| Qwen R2 final adapter | 36.28% (418) | 34.90–37.76% | 33.33% (64) |
| Intern-Decision-0.8B | 58.16% (670) | 54.86–61.63% | 50.00% (96) |
| Kev-0.8B | 76.13% (877) | 70.83–81.25% | 74.48% (143) |

The intervals resample the 64 source groups, keeping each group's three records and six
permutations together. They use 2,000 paired source-group bootstrap draws, seed
`20261006`, with 95% percentile intervals. The paired all-order difference for the R2
adapter minus Qwen base was **−2.52 percentage points** (95% interval **−4.25 to
−0.87 pp**). The adapter minus Intern difference was **−21.88 pp** (−25.87 to −18.14),
and adapter minus Kev was **−39.84 pp** (−44.79 to −34.81).

Permutation behavior needs to be read with accuracy. Qwen base changed its semantic
prediction on 192/192 records and had zero records correct under all six orders. The R2
adapter changed its prediction on 65/192 records and was correct under all six orders on
62/192; on the original order it predicted **entailment for all 192 records**, which is
only 64/192 correct. Thus fewer flips alone do not show useful stability: an unchanged
wrong answer is stable too. Intern changed 139/192 records and was correct under all six
orders on 43/192; Kev changed 20/192 and was correct under all six on 137/192. The
canonical summary includes exact ties, unique-maximum coverage, per-class recall, and
all-order and original-order confusion matrices.

## Interpretation

This is conditional evidence on one 192-record, 64-group development panel, not a sealed
test or a population estimate. The reference systems use FP32 weights; Qwen uses a BF16 base and an FP32 adapter.
The systems also use different prompt and tokenizer paths; SNLI schema familiarity and pretraining exposure remain possible. The panel uses
different examples from R2, so this comparison cannot isolate the effect of balancing
from the effect of changing examples.

The earlier R2 preregistered SNLI non-regression gate remains failed. R1 does not repair
or replace that gate, establish generalization, or support a broad model-superiority
claim. The candidate synthetic training set remains untrained and is not part of these
results.

## Reproducibility and operations

The passed receipt is
[`snli-balanced-2026-10-04-r1-receipt.json`](../artifacts/snli-balanced-2026-10-04-r1-receipt.json)
(SHA-256 `76be5d7870a8780e5bebb59adf4fb834c908042b6686a21fdc045965ab68eed7`); its
analyzed output is
[`snli-balanced-2026-10-04-r1-analysis.json`](../artifacts/snli-balanced-2026-10-04-r1-analysis.json)
(SHA-256 `668d119255e017e3b1a42e31e378aa6fb15c2ba2d8731df6e767dd97eba045b1`). The
receipt pins protocol SHA-256
`e41334b9d2f6b47d057907782c889251f8e952fae9e77d173e1d7bd69062a4ae`; Modal SDK was
1.6.1. Independent recounts of every output, confusion-relevant prediction, flip, tie,
all-six correctness count, and paired interval matched the analysis.

The exact-source runner commit `563e8e362bcca3180f12e6dd96048758e8af5df3` passed main
CPU CI run `37264163333`, including lint, formatting, typecheck, 323 tests, and the
installed-CLI smoke test. After the run, Modal app `ap-XQ4OLLSz9D09DWnkaJ2bmv` was
stopped with zero tasks at 2026-10-04 21:40:07 PDT. The teardown artifact records a
workspace-wide current-month snapshot of `$0.75366943` metered and `$0` billed after
credits. It is not a per-run invoice and cannot be attributed to R1.

See the machine-readable [verification summary](verification/snli-diagnostic-summary.json)
for source hashes, aggregate metrics, independent-validation evidence, and operations
details.
