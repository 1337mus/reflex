# Reflex

Reflex is in a research and design stage. The immediate question is whether a small open model can make runtime-defined decisions from one forward pass, and whether that approach adds measurable value beyond existing decision models. **No Reflex model has been trained or benchmarked.**

Start with the [research plan](docs/research-plan.md) and [runtime decisions ADR](docs/adr/0001-runtime-decisions.md). Intern-Decision-0.8B already documents candidate-symbol logits from runtime schemas in one Qwen3.5 forward pass; Kev-0.8B documents a Qwen3.5-0.8B pointer head, calibration, and task-held-out evaluation. The first experiment is a fair comparison against these systems before Reflex training. Reflex will need a measured improvement or a rigorous comparative result to justify a distinct claim.

The current proposal is a narrow English v0.1: single-label decisions over 2–16 runtime options, with an explicit abstention signal, fixed input limits, and remote-only GPU experiments. These are design choices, not shipped capabilities or measured results. See the plan for scope, source links, implementation gates, and spend envelopes.
