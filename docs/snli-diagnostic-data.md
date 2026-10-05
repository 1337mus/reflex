# Balanced SNLI diagnostic data

**Prepared:** 2026-10-04. This is a separate, development-only diagnostic panel. It does
not replace the failed R2 learning gate or establish population accuracy.

## Source and exclusions

The panel uses only the Stanford SNLI 1.0 development member. The source archive SHA-256 is
`afb3d70a5af5d8de0d9d81e2637e0fb8c22d1235c2749d83125ca43dab0dbd3e`; the extracted
development JSONL SHA-256 is
`9c03faff70182ef086ebfeed2cffbabb5fcc6a84a8b3314decbbb5b01f07f4bf`. Preparation verifies
that the separate development file equals the member in the pinned archive. It reads no
train or test member.

Before selection, the recipe excludes all source groups in two byte-pinned recipes:

- `data/baselines/broader-dev-recipe.json`, SHA-256
  `6450653a445cc12b21304c267cdf0c8976009a01c6982fb07751b349cd81f68b` (32 groups).
- `data/pilots/real-pilot-v1-recipe.json`, SHA-256
  `2315f8c39efa95f2cc018e8d5a7281ba37f304f66161aa737a6ab9389bdded4f` (128 groups).

The recipes contain 160 unique, disjoint groups, all present in the parsed development
source. The established SNLI parser preserves caption/premise grouping, normalized-pair
deduplication, and its error on conflicting normalized labels. It yields 3,319 source
groups; after exclusions, 2,488 groups contain all three labels.

## Selection and artifacts

The recipe seed is `reflex-snli-balanced-v1:20261006`. The 64 selected group IDs are ranked
by ascending SHA-256 of `seed:group:{source_group_id}`, then by group ID. Within each group,
one pair per label is selected by ascending SHA-256 of `seed:pair:{pairID}`, then pair ID.
Selection is independent of input row order and model output. It does not choose the raw
lexical-minimum pair ID.

The result has 192 records: 64 entailment, 64 neutral, and 64 contradiction, with all three
labels represented once in each of 64 source groups. Records keep the source group ID and
the established SNLI question, option order, option descriptions, and answer IDs. The
manifest identifies `snli-balanced-v1-development` as a benchmark development dataset from
`stanford-snli-1.0`; it declares no held-out task family.

The locally generated, ignored candidate is in `data/processed/snli-balanced-v1/`:

| File | SHA-256 |
| --- | --- |
| `records.jsonl` | `a1fa41d19b381e227ebca258b1561e7e39184f6f58a325aaa5a2b1ff61ddd98c` |
| `manifest.json` | `4ecab09c6f013c274b38a81c6ef226226fee6520745e01457f461c5e5dc628bd` |
| `recipe.json` | `e11fc94b99376f4860caf3b9afbb22026e6015ee7b4bfeb42344a7a0cc0bbd58` |

The recipe records source and prior-recipe hashes, exclusion IDs, group counts, selected
group and pair IDs, labels, native development row references, and output hashes. It
contains no premise or hypothesis text. The data source is licensed under CC BY-SA 4.0;
consult the [Stanford SNLI notice](https://nlp.stanford.edu/projects/snli/) and the local
[`SNLI-DATA-NOTICE.txt`](../data/notices/SNLI-DATA-NOTICE.txt).

The exact [manifest](../data/diagnostics/snli-balanced-v1-manifest.json) and
[recipe](../data/diagnostics/snli-balanced-v1-recipe.json) are versioned. Corpus text
remains in the ignored processed directory.

Run the preparation CLI in plan-only mode by default. `--write` requires a new output
directory and refuses to overwrite any existing path. Raw and processed data remain
ignored. No model inference, training, or paid service was used to prepare this panel.

## Interpretation boundary

This is a conditional comparison panel drawn from SNLI development data, not a population
sample or a sealed evaluation. Its 192 records form 64 dependent source groups. A later
evaluation should compare models within this fixed panel and preserve groups in uncertainty
resampling. Comparing its score with the R2 SNLI score cannot isolate the effect of
balancing because the selected examples also differ. The panel cannot repair the preregistered
R2 failure or support a broad superiority claim. If it informs future training or model
selection, later measurements on it are adaptive development evidence.
