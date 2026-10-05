# BoolQ source amendment, version 1

Recorded 2026-10-04 before any model scores under the
[two-task protocol](broader-baseline-protocol.md).

The original BoolQ README's Google Cloud download redirects to sign-in. The
equivalent public `storage.googleapis.com/boolq/dev.jsonl` endpoint returns
`UserProjectAccountProblem`, reporting that the owning project's billing
account is closed. No requester billing project or credentials were supplied.

Use the public [google/boolq mirror at commit
35b264d03638db9f4ce671b711558bf7ff0f80d5](https://huggingface.co/datasets/google/boolq/tree/35b264d03638db9f4ce671b711558bf7ff0f80d5).
Its README identifies the original BoolQ dataset, CC BY-SA 3.0 license, and
3,270-row `validation` split. Only that development split is acquired:
`data/validation-00000-of-00001.parquet`, 1,257,630 bytes, SHA-256
`52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8`.
The downloaded bytes match the mirror's published LFS SHA-256.

This representation contains `question`, `answer` and `passage`, with no page
title. Group its rows by normalized passage; no missing title can provide a
grouping link. Consequently, distinct passages from the same unknown page can
remain separate groups. Do not claim page-level independence or byte identity
with the unavailable original JSONL. Record the mirror commit, raw file hash,
conversion and prepared-record hashes in the reproducible data recipe.

The sample size, label-independent selection, prompts, model pins, answer
permutations, metrics and training separation are unchanged. The original
protocol is preserved. Both documents must be present and hash-bound in the
run receipt.
