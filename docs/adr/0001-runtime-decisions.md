# ADR 0001: Runtime decision contract and first inference path

- **Status:** Proposed; research-only, 2026-10-04
- **Scope:** Reflex v0.1 design and first experiment
- **Decision owners:** Project owner and research lead

## Context

Reflex is intended to choose among options supplied at inference time, with one model pass and no generated explanation or JSON. The preferred starting checkpoint is Qwen/Qwen3.5-0.8B-Base. This is not an established performance choice: its quality, speed, calibration, and compatibility with the chosen training stack remain unmeasured here.

Close prior art materially changes the project thesis. [Intern-Decision-0.8B](https://huggingface.co/internlm/Intern-Decision-0.8B) documents runtime schemas, candidate-symbol logits at JSON placeholders, and calibrated one-pass Qwen3.5 decisions. [Kev-0.8B](https://huggingface.co/jaredpalmer/kev-0.8b) uses the same Qwen3.5-0.8B-Base family with a learned option pointer head and reports calibration, whole-task-family evaluation, and option-order sensitivity. Candidate logits, runtime schemas, calibration, and family-held-out evaluation cannot be presented as Reflex's novelty without further evidence. Published results remain claims by their authors until reproduced on a shared protocol.

## Decision

1. Keep v0.1 narrow: English text, one mutually exclusive answer, and 2–16 options. Binary decisions are the two-option case. Reject oversized inputs rather than silently truncating them. Start at 2,048 input tokens; treat 4,096 as a later stress test.
2. Represent option IDs separately from semantic option content. Score only options valid under the current schema. The proposed baseline is the final real prompt-token hidden state projected onto verified single-token option symbols, then softmax over those option scores. Map symbols back to stable semantic IDs; never infer semantics from answer position.
3. Require each option symbol to be a single tokenizer token in its actual concatenated prompt boundary. Establish whether a text-only causal-model path loads the pretrained checkpoint completely before any training.
4. Keep a semantic “unknown/none/insufficient evidence” answer, when appropriate to the task, inside that task's legal option set. Keep epistemic abstention separate as an abstained boolean, a reason code, and a nullable answer. Candidate probabilities are conditional on the supplied legal options; they are not automatically calibrated probabilities of correctness.
5. Do not adopt cached-prefix serving, arbitrary option counts, multi-label decisions, ranking, scalar regression, RL, or a 4B training run in v0.1. A dynamic pointer head is a later hypothesis for larger option sets or improved robustness, not an assumed invariant.
6. Before Reflex training, reproduce the closest existing decision models—Intern-Decision-0.8B's candidate-symbol path and Kev-0.8B's pointer path—on a pre-registered, leakage-resistant suite. Proceed with training only if a distinct empirical target remains and those implementations can be compared fairly. A plausible target is improved option-order stability and selective risk/coverage on untouched task families, but that target is a hypothesis, not a uniqueness claim.

## Consequences and verification

This contract keeps the first system measurable and limits prompt and output ambiguity. The symbol path is valid only if pretrained weights load without missing/uninitialized text weights, option-symbol tokenization is correct at the concatenated boundary, semantic mappings survive option shuffles, and batch and single-example scores agree within a documented tolerance. Compare candidate-only projection against the model's native output projection within a documented numerical tolerance before optimizing it. Intern-Decision scores allowed symbols immediately before placeholders in a JSON skeleton; Kev scores option-closing states against the final question state. Treat both as explicit baselines.

The one-shot Modal compatibility smoke passed on 2026-10-04: Qwen text-model loading, semantic mapping, and single/batch parity passed on three synthetic prompts. The [receipt](../verification/qwen-modal-smoke.json) records a maximum logit difference of 0.125 (the configured limit) and a maximum probability difference of 0.00419. This is compatibility evidence only; it does not establish accuracy, calibration, optimized-kernel support, or performance.

Qwen3.5 is a hybrid architecture. Shared-prefix reuse is deferred: recurrent/convolution state and full-attention KV state may both need independent forks per suffix. Any later cache implementation must show state isolation and cached-versus-uncached equivalence. Pointer scoring against per-option marker states may be useful but is a trainable design hypothesis; one causal pass alone does not guarantee order invariance.

The model family is described as MoE in broader marketing, but the 0.8B text configuration is a dense text model. The Hugging Face parameter total includes the vision tower and should not be reported as an exact text-parameter count. Use “Qwen3.5-0.8B family” unless counting a specific text-only module. Relevant references: [checkpoint config](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/blob/main/config.json) and [Transformers Qwen3.5 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5).

## Revisit when

The ADR can be revised after the prior-art comparison, model-load compatibility smoke test, and first scored evaluation. Expand the schema only when an identified task family requires it and its metrics and output contract are specified. Treat any claim of generality, calibration, position robustness, or advantage as a result to measure, not a premise.
