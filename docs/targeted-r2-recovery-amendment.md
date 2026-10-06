# Targeted R2 saved-result recovery amendment (revision 2)

**Current revision: 2, dated 2026-10-06.** This amendment has two distinct
stages. Revision 1 defined an operational admission route before the first
production calculation. Revision 2 corrects only the independent checker's
Unicode semantic request hashing, after that calculation had already run.

## Revision 1 — initial operational amendment before calculation

The initial amendment was dated 2026-10-05. Its signed source commit was
`76ad2f97619d118110d61f67c46d8d4417f0421f`, and its verification metadata
SHA-256 was
`2b3c099e41fd07af2e9b622187cc622f20f33fdaa2ea768a43f4699d198dd328`.
The V1 documents and evidence are preserved under
`artifacts/targeted-r2-recovery-v1-evidence/`.

V1 established a separate operational acceptance route for the already-saved
worker results after the controller stopped before writing a complete success
receipt. It did not revise the original protocol, receipt, training, data,
metric rules, or learning thresholds. The first production calculation was
made under V1, before the checker correction described below.

## Revision 2 — checker-only correction after calculation

The first V1 production calculation has already completed, and its official
arithmetic succeeded. The subsequent independent recount failed at the
Unicode semantic request-hash comparison. Therefore this correction is
explicitly **after the first production calculation**; V2 must not be
described as a pre-calculation fix or as having governed that calculation.

V2 changes only the recovery wrapper's in-memory semantic `request_hash`
adapter, using the producer-compatible UTF-8 JSON encoding with
`ensure_ascii=False`. The original independent checker,
`.context/recount_targeted_study.py`, remains byte-identical. Its presentation
order requirement stays intact: request option IDs must still equal
`order_ids`. The correction does not loosen equality, disable hash checking,
or change the generic canonical encoding used for receipt, plan, and result
pins.

Synthetic tests check Unicode and non-BMP strings, optional descriptions,
ASCII compatibility, option permutation invariance for semantic hashing,
rejection of mismatched presentation order, and rejection of changed request
text or hashes. They also check that the generic canonical encoding and
frozen study checker stay unchanged. Separately, a count-only check of the
6,182 saved presentations validates presentation counts, hashes, and order
IDs; it did not read answers, predictions, or scores, or compute metrics.
This amendment does not claim that a V2 recount or V2 analysis has completed.

## Operational admission contract carried forward from V1

The amendment accepts complete, uniquely selected **recorded worker results**
after the recovery program checks the original receipt and complete event log,
monotone role-call and result snapshots, source and input bytes, bundle-file
hashes, role payloads, and all three raw results. Each result is checked again
by the frozen per-worker validator. Existing saved validations must match
their raw results; missing validations are reconstructed and attributed to
this recovery check. The recorded stopped-app evidence, zero-task
observations, and saved adapter path and size metadata must agree. Adapter
metadata is not a fresh hash of the saved adapter bytes.

Answer loading and analysis remain blocked until a separate clearance binds
the revision 2 amendment hash and the current signed source commit, passed CI
on that same commit, root and independent approvals over the reviewed file
hashes, and a passing synthetic-only independent-checker result. Admission
and the frozen receipt gate both run before the host answer loader. Analysis
reuses the frozen metrics and all 24 learning checks. The recovery report
identifies the original failure, recovery admission, worker revalidation,
and provenance separately; it does not emit a generic execution-success
flag. New output is written to a new, exclusive directory, leaving the
original evidence unchanged.

## Preserved limits and scientific scope

The original controller status remains **failed** because the controller
stopped with `KeyboardInterrupt` before writing a complete success receipt.
The V2 checker correction does not change any study data, saved worker
outputs, source/data/result pins, metric or bootstrap rules, aggregation,
learning thresholds, or presentation-order requirements. Any V2 production
replay must preserve the V1 metrics and learning objects exactly before an
independent agreement claim is made.

This exception does not prove that the provider made no additional attempts
or physical forwards; provider-wide exactly-once execution remains uncertain.
The cause and interrupted stage are unknown. It also does not claim fresh
adapter byte integrity, multiple-seed robustness, or unseen-template or
pretraining exposure. The existing sampling and scientific limitations
remain in force. No retry, training, or scoring is part of admission.
