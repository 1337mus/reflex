# Private SNLI training candidate

This preparation creates a private, train-only candidate of 500 natural-language inference examples. Each example presents a **premise** and a **hypothesis** (a claim). The model chooses **entailment** when the premise guarantees the claim, **neutral** when it neither guarantees nor contradicts the claim, and **contradiction** when it contradicts the claim. The candidate is data preparation only, not a model result or release approval.

The pinned archive contains 550,152 train rows and 10,000 development rows. The parser keeps unlabeled rows and conflicting pairs as source-link evidence, while only consistent labeled pairs can become candidate examples.

| Source measure | Train | Development |
| --- | ---: | ---: |
| Rows | 550,152 | 10,000 |
| Unlabeled rows | 785 | 158 |
| Conflicting normalized pairs | 66 | 0 |
| Labeled rows in conflicting pairs | 151 | 0 |
| Consistent labeled pairs | 548,648 | 9,840 |
| Connected source groups | 150,686 | 3,319 |

Groups are connected components over normalized premise anchors and normalized caption IDs. Anchors are type-labeled as `premise` or `caption`, so an identical string in the two namespaces does not link rows. A conflicting premise/hypothesis pair is excluded as an example, but its anchors remain in the graph. Unlabeled rows likewise contribute anchors and never become examples.

The builder parses the complete development member, then excludes each whole train group that shares any premise or caption anchor with any development row. There are 16 shared premise anchors and no shared caption anchors; this removes 16 train groups and 78 candidate pairs. One exact normalized pair occurs across the source splits and is among the exclusions. After filtering, 548,570 pairs remain across 150,658 groups; 147,388 groups contain candidates for all three labels.

Selection uses seed `20261008`. It ranks eligible groups by SHA-256 of the seed and group ID, assigns quotas of 167 entailment, 167 neutral, and 166 contradiction groups using a separate group ranking, then ranks pairs within each assigned group and label using a separate hash of the pair ID. It emits records in group-rank order. The precise selection and source pins are in the [source-only recipe](../data/training/snli-v1-recipe.json); the [train manifest](../data/training/snli-v1-manifest.json) declares the split. Neither file contains premise or hypothesis text. The raw candidate records remain local under ignored `data/processed/snli-training-v1/`.

The builder verifies the SNLI 1.0 archive, train member, and full development member against their pinned SHA-256 values. It inventories archive member names but opens only the train and development JSONL members; it does not open the official test member. See the [verification summary](verification/snli-training-data-summary.json) for source, output, and code hashes, exact counts, and validation evidence.

## Reproduce

Run a plan to verify the pinned source and display the deterministic output hashes without writing files:

```sh
uv run --offline --locked python -m experiments.prepare_snli_training \
  --archive data/raw/snli_1.0.zip
```

To write a candidate, choose a new, unused output directory and evidence path; the writer fails if either already exists:

```sh
uv run --offline --locked python -m experiments.prepare_snli_training \
  --archive data/raw/snli_1.0.zip \
  --write \
  --output-dir data/processed/snli-training-v1-check \
  --evidence .context/snli-training-v1-check.json
```

The dataset is [SNLI 1.0](https://nlp.stanford.edu/projects/snli/), introduced by Bowman, Angeli, Potts, and Manning in [“A Large Annotated Corpus for Learning Natural Language Inference”](https://aclanthology.org/D15-1075/). The dataset is listed as [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

The SNLI project attributes most premises to [Flickr30k captions](https://shannon.cs.illinois.edu/DenotationGraph/) and a smaller portion of the training split to [Visual Genome](https://visualgenome.org/). Preserve the Stanford and upstream attribution and license notices. Review rights separately before distributing trained weights or adapters.

After this training, our existing SNLI panels will measure performance within the training source. They will not measure transfer to a new task family. Base-model pretraining exposure is unknown.
