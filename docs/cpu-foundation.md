# CPU foundation

This package is an offline data, scoring, and evaluation layer. It does not load a model, generate predictions, fit calibration parameters, train weights, or require a GPU. Predictions are supplied as saved logits. The included fixture is synthetic and is useful only for exercising the formats and commands.

One narrow external preflight checked the public Qwen3.5-0.8B-Base tokenizer JSON at revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`: the newline-ending suffix `\nAnswer:\n` tokenized all 16 `A`–`P` candidates singly, while the trailing-space suffix `\nAnswer: ` did not. The reproducibility receipt is [qwen-tokenizer.json](verification/qwen-tokenizer.json); it records `tokenizers` 0.23.2, the tokenizer JSON SHA-256, option counts 2 and 16, encoded prompt lengths 54 and 256, and candidate token IDs 32–33 and 32–47 respectively. This establishes only those tokenizer results. It does not verify model weights loading, prompt execution, inference, batching, or end-to-end Qwen compatibility.

## Files and joins

The manifest is one strict JSON object. `data_kind` is `fixture` or `benchmark`; `datasets` lists each `dataset_id`, its `source_id`, `task_family`, `split`, and source URI, revision, and license. A split is `train`, `development`, `calibration`, or `test`. `held_out_families` is optional and declares families that must appear only in `test`.

Records may be a JSON array (`.json`) or one JSON object per non-empty line (`.jsonl`). Each record contains a unique `record_id`, `dataset_id`, `source_group_id`, `answer_id`, and a `request`. A request has nonblank `context` and `question` plus 2–16 `options`; every option has a unique nonblank `id` and `label`, with an optional `description`. The gold `answer_id` must be one of the request's option IDs.

The request's `schema_hash` covers its question and options, while `request_hash` also includes the context. Both hashes canonicalize option order by ID, serialize UTF-8 JSON with sorted object keys and compact separators, and use SHA-256. The option labels and descriptions are part of the hash; reordering otherwise identical options does not change it.

Predictions may also be a JSON array or JSONL. Each entry has a `record_id`, a `request_hash`, a nonempty `logits` object mapping every legal option ID to one finite number, and a `model_revision`. The `request_hash` is the record's 64-character lowercase SHA-256 semantic request hash and must match the request in the records file. Evaluation joins by `record_id`, not row order; it requires exactly one prediction per record, the matching request hash, the exact option-ID set, and one shared model revision. All files reject unknown fields and duplicate JSON object keys. The report records SHA-256 hashes of the original input file bytes, so reformatting a file changes its hash.

Example prediction entry:

```json
{"record_id":"support-1","request_hash":"4da573e7f0acf6b67f26bd5b319d508a2b86b9f7af4398e0f21ffdd1829d3e61","logits":{"billing":3.0,"access":0.0,"fraud":0.0},"model_revision":"synthetic-logits-v1"}
```

`validate` checks record structure, dataset references, and declared split lineage. `evaluate` audits the full supplied manifest and then scores records from exactly one split per report. Keep separate reports for train, development, calibration, and test; a report cannot mix splits.

```sh
uv run reflex validate \
  --records data/fixtures/records.jsonl \
  --manifest data/fixtures/manifest.json

uv run reflex evaluate \
  --records data/fixtures/records.jsonl \
  --manifest data/fixtures/manifest.json \
  --predictions data/fixtures/predictions.jsonl \
  --output /tmp/reflex-fixture-report.json
```

The evaluator applies temperature scaling before softmax: each logit is divided by the positive `--temperature` value, then normalized over the supplied options. The default temperature is `1.0`. Exact ties for the highest scaled logit always abstain; `top_option_id` is still set to the lexicographically smallest tied ID for diagnostics and accuracy. Otherwise, `--min-confidence` abstains when the top probability is strictly below the threshold; a probability equal to the threshold is accepted. Probabilities are conditional on the options provided, and are not automatically calibrated probabilities of correctness.

`--calibration-id` is a caller-supplied label. The report records `profile_id_supplied` when it is present and `uncalibrated` otherwise. It does not load or verify a calibration profile. The evaluator applies, but does not fit or select, temperature or confidence thresholds; it also does not prove that those values came from a calibration split. Select calibration parameters on designated development/calibration data, freeze them, and evaluate a separate test split once. Keep the calibration data and procedure in your own provenance record.

## Metrics

All metrics are computed per dataset and, where applicable, pooled across rows. Per-dataset macro-F1 averages label F1 over the union of gold labels and predicted top-option IDs in that dataset, using zero when a label has no true positives. `micro_accuracy` is pooled top-option accuracy. It counts the diagnostic top option even when the policy abstains; coverage and accepted risk report abstention separately.

For each example, NLL is the negative stored log-softmax value for the gold option. It uses the computed `log_probability`, not a clipped probability. Multiclass Brier score is the sum over options of `(p - y)²` for each example, then averaged over examples; it does not divide by the number of classes. ECE uses the maximum option probability as confidence and top-option correctness as accuracy. It has ten left-closed/right-open bins `[0.0,0.1)` through `[0.9,1.0]`; probability 1.0 is included in the final bin. ECE is the sum of each nonempty bin's share of examples multiplied by the absolute gap between mean confidence and mean accuracy in that bin.

Overall `macro_dataset_accuracy`, `macro_dataset_f1`, `macro_dataset_nll`, `macro_dataset_brier`, and `macro_dataset_ece` are unweighted averages of the corresponding per-dataset metric, so each dataset contributes equally. `pooled_nll`, `pooled_brier`, and `pooled_ece` are computed across all rows and therefore weight datasets by their record counts. Overall `micro_accuracy` is likewise pooled.

Coverage is the fraction of records the frozen policy accepts. Accepted risk is the fraction of accepted records whose top option is wrong; it is `null` when no examples are accepted. The risk-coverage curve is a diagnostic over all records, sorted by maximum option probability, independent of the confidence threshold. It starts at coverage 0 with null risk and adds groups of exactly tied confidence together. At each right endpoint, risk is cumulative top-option error rate. AURC uses the right-endpoint rectangle convention: for each group, add the coverage increase multiplied by the risk at that group's endpoint. It is not trapezoidal interpolation.

## Split lineage and leakage limits

The manifest audit rejects a `source_id` assigned to more than one split, a `(source_id, source_group_id)` appearing in more than one split, and an exact semantic request hash appearing in more than one split. The request hash includes context, question, and options (including labels/descriptions) in option-ID order, so option order alone does not make a different request. A declared held-out family must exist in the manifest and occur only in `test`. Reports include the evaluated split.

These checks enforce declared IDs and exact request matching; they do not establish that the metadata is truthful or that a dataset is uncontaminated. They do not find paraphrases, near-duplicates, related source content, benchmark or pretraining contamination, or overlap introduced before records were assembled. They cannot establish that model training, prompt selection, or calibration stayed away from test data. Maintain source provenance and perform content-level deduplication and protocol review separately. The synthetic fixtures provide no evidence about any model.

## Glossary

- **Weights:** learned numeric parameters stored by a model.
- **Logits:** unnormalized numeric scores for candidate options; softmax turns them into a distribution over those options.
- **Softmax:** exponentiates and normalizes scores so probabilities sum to one.
- **LoRA (Low-Rank Adaptation):** compact trainable adapter matrices added over a frozen base model. LoRA can reduce the number of trainable parameters; it does not automatically make inference faster. This CPU foundation does not train or run LoRA models.
