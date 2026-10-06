# Targeted R2 saved-result recovery amendment

**Dated 2026-10-05.** This amendment defines a separate operational acceptance
route for the saved worker results from `targeted-reasoning-2026-10-05-r2`.
It applies only to the run, plan, receipt, source snapshot, role payloads, call
identities, and evidence hashes fixed in
`docs/verification/targeted-r2-recovery-amendment.json`. It does not revise the
original protocol or receipt. The original controller status remains **failed**
because the controller stopped with `KeyboardInterrupt` before writing a complete
success receipt.

The amendment accepts complete, uniquely selected **recorded worker results**
after the recovery program checks the original receipt and complete event log,
monotone role-call and result snapshots, source and input bytes, bundle-file
hashes, role payloads, and all three raw results. Each result is checked again by
the frozen per-worker validator. Existing saved validations must match their raw
results; missing validations are reconstructed and attributed to this recovery
check. The recorded stopped-app evidence, zero-task observations, and saved
adapter path and size metadata must agree. Adapter metadata is not a fresh hash of
the saved adapter bytes.

Answer loading and analysis remain blocked until a separate clearance binds this
amendment hash and the current signed source commit, passed CI on that same
commit, root and independent approvals over the reviewed file hashes, and a
passing synthetic-only independent-checker result. Admission and the frozen
receipt gate both run before the host answer loader. Analysis reuses the frozen
metrics and all 24 learning checks. The recovery report identifies the original
failure, recovery admission, worker revalidation, and provenance separately; it
does not emit a generic execution-success flag. New output is written to a new,
exclusive directory, leaving the original evidence unchanged.

This exception does not prove that the provider made no additional attempts or
physical forwards; provider-wide exactly-once execution remains uncertain. The
cause and interrupted stage are unknown. It also does not claim fresh adapter
byte integrity, multiple-seed robustness, or unseen-template or pretraining
exposure. The existing sampling and scientific limitations remain in force.
No retry, training, or scoring is part of admission.
