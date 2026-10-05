# Tool-choice data v1 release

**Status (2026-10-05): generated and independently verified.** The release contains 112 synthetic decision records in 112 distinct tool catalog groups. An independent verifier recomputed all 112 answers from the visible prompt facts. No model has been scored or trained on these data.

## Release contents

| Split | Records | Purpose |
| --- | ---: | --- |
| Train | 56 | Training |
| Development | 14 | Find errors before the final test |
| Calibration | 14 | Check confidence scores |
| Test (sealed) | 28 | Final evaluation after the plan is fixed |
| **Total** | **112** | |

| Situation | Records |
| --- | ---: |
| Known permissions; a tool qualifies | 16 |
| Known permissions; no tool qualifies | 16 |
| One permission is missing; the same tool wins either way | 16 |
| One permission is missing; no tool qualifies either way | 16 |
| One permission is missing; different tools may win | 16 |
| One permission is missing; a tool or no tool may result | 16 |
| Two permissions are missing | 16 |

Menus have 4, 6, or 8 options: 2, 4, or 6 tools plus the two fixed answers. Full menu and answer aggregates are in the verification summary.

## Verification and provenance

One fresh 64-bit seed, `4149131584908688448`, was selected after source review and exact-commit CI passed. It was selected once; no seed search was performed.

The generator [source commit](https://github.com/1337mus/reflex/commit/c1de76ccbdf0c91701a558148d978a8d63926b82) passed [CI](https://github.com/1337mus/reflex/actions/runs/37292615697). The audit [source commit](https://github.com/1337mus/reflex/commit/f51c65314643b6ee24f0b40170defb87bf9a0b54) passed [CI](https://github.com/1337mus/reflex/actions/runs/37292560896).

The independent release verifier passed. Clean verification passed 677 tests, Ruff, formatting, mypy, and fixture CLI validation and evaluation. The release recipe pins 12 source files.

The [verification summary](verification/tool-data-summary.json) contains full source and export hashes, split aggregates, seed-selection evidence, and check details. The released manifest and recipe are copied byte-for-byte to [tool-v1-manifest.json](../data/training/tool-v1-manifest.json) and [tool-v1-recipe.json](../data/training/tool-v1-recipe.json).

```mermaid
flowchart LR
    A[Reviewed source and successful exact-commit CI] --> B[One fresh seed, no search]
    B --> C[112 records across four splits]
    C --> D[Independent answer and bundle verification]
    D --> E[Next: fixed matched continued-training study]
    E --> F[Equal presentations and updates, then retention checks]
```

The final review also exercised five full-size toy seeds; those were generator checks, not release test runs. The 28 sealed test records are useful for a small, fixed evaluation but are too few to support broad performance conclusions. These synthetic tasks do not establish broad reasoning ability, and no model result exists yet.
