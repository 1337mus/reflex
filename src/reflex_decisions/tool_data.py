"""Deterministic CPU-only tool-choice fixture generation."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections.abc import Iterator
from dataclasses import dataclass, replace

from .data import DatasetSpec, DecisionRecord, SplitManifest, SplitName, audit_splits
from .rendering import render_prompt
from .tool_data_audit import audit_tool_data, classify_scenario
from .tool_data_identity import split_for_structure, structural_catalog_hash
from .tool_data_spec import (
    CASE_KINDS,
    FAMILY,
    MAX_COUNTS,
    SOURCE_ID,
    SPLITS,
    TEMPLATE_IDS,
    VERSION,
    VOCABULARIES,
    ToolDatum,
    ToolVocabulary,
)
from .tool_rules import INSUFFICIENT_INFORMATION, NO_ELIGIBLE_TOOL, ToolScenario, ToolSpec, solve
from .tool_text import render_tool_request
from .tool_text_parser import parse_tool_prompt, verify_tool_request

_SOURCE_URI = "synthetic://reflex/tool-choice/v1"
_LICENSE = "self-authored synthetic"
_MAX_CATALOG_ATTEMPTS = 4096
_TOOL_COUNTS = (2, 4, 6)
_PERMISSION_VALUES = (False, True, None)


@dataclass(frozen=True, slots=True)
class ToolGenerationConfig:
    """Immutable seed and per-split sizes for one tool fixture candidate."""

    seed: int
    counts: tuple[int, int, int, int] = (56, 14, 14, 28)

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an integer between 0 and 2**64 - 1")
        if (
            type(self.counts) is not tuple
            or len(self.counts) != len(MAX_COUNTS)
            or any(
                type(count) is not int or not 1 <= count <= maximum
                for count, maximum in zip(self.counts, MAX_COUNTS, strict=True)
            )
        ):
            raise ValueError("counts must be a four-item tuple within the per-split maxima")


@dataclass(frozen=True, slots=True)
class ToolCandidate:
    config: ToolGenerationConfig
    manifest: SplitManifest
    rows: tuple[ToolDatum, ...]


def generate_tool_data(config: ToolGenerationConfig) -> ToolCandidate:
    """Build and audit a deterministic in-memory tool-choice candidate."""
    if not isinstance(config, ToolGenerationConfig):
        raise TypeError("config must be a ToolGenerationConfig")
    manifest = _manifest()
    dataset_ids = {split: f"{VERSION}-{split}" for split in SPLITS}
    seen_catalogs: dict[str, set[str]] = {split: set() for split in SPLITS}
    rows: list[ToolDatum] = []

    for split, count in zip(SPLITS, config.counts, strict=True):
        for index in range(count):
            tool_count = _TOOL_COUNTS[(index + index // 21) % len(_TOOL_COUNTS)]
            case_kind = CASE_KINDS[index % len(CASE_KINDS)]
            multiple_outcome = ("tool", "none", "conflict")[(index // 7) % 3]
            datum = _generate_row(
                config,
                split,
                index,
                tool_count,
                case_kind,
                multiple_outcome,
                dataset_ids[split],
                seen_catalogs[split],
            )
            rows.append(datum)

    frozen_rows = tuple(rows)
    audit_tool_data(frozen_rows, manifest)
    audit_splits(manifest, tuple(row.record for row in frozen_rows))
    return ToolCandidate(config, manifest, frozen_rows)


def _manifest() -> SplitManifest:
    return SplitManifest(
        data_kind="fixture",
        datasets=tuple(
            DatasetSpec(
                dataset_id=f"{VERSION}-{split}",
                source_id=SOURCE_ID,
                task_family=FAMILY,
                split=split,
                source_uri=_SOURCE_URI,
                source_revision=VERSION,
                license=_LICENSE,
            )
            for split in SPLITS
        ),
        group_partitioned_sources=(SOURCE_ID,),
    )


def _rng(config: ToolGenerationConfig, split: str, index: int, stream: str) -> random.Random:
    encoded = json.dumps(
        (VERSION, config.seed, split, index, stream), separators=(",", ":")
    ).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(encoded).digest(), "big"))


def _base_scenario(vocabulary: ToolVocabulary, rng: random.Random, tool_count: int) -> ToolScenario:
    capabilities = tuple(rng.sample(vocabulary.capabilities, len(vocabulary.capabilities)))
    input_types = tuple(rng.sample(vocabulary.input_types, len(vocabulary.input_types)))
    permissions = tuple(rng.sample(vocabulary.permissions, len(vocabulary.permissions)))
    tool_ids = rng.sample(vocabulary.tool_ids, tool_count)
    costs = rng.sample(range(101), tool_count)
    tools = []
    for tool_id, cost in zip(tool_ids, costs, strict=True):
        tools.append(
            ToolSpec(
                id=tool_id,
                capabilities=tuple(rng.sample(capabilities, rng.randint(1, len(capabilities)))),
                input_types=tuple(rng.sample(input_types, rng.randint(1, len(input_types)))),
                required_permissions=tuple(
                    rng.sample(permissions, rng.randint(0, len(permissions)))
                ),
                cost=cost,
            )
        )
    rng.shuffle(tools)
    return ToolScenario(
        capabilities,
        input_types,
        permissions,
        tuple(tools),
        (capabilities[0],),
        input_types[0],
        (False,) * len(permissions),
    )


def _query_candidates(
    base: ToolScenario, case_kind: str, multiple_outcome: str
) -> Iterator[ToolScenario]:
    for capability_count in range(1, len(base.capabilities) + 1):
        for required in itertools.combinations(base.capabilities, capability_count):
            for input_type in base.input_types:
                for permission_values in itertools.product(
                    _PERMISSION_VALUES, repeat=len(base.permissions)
                ):
                    scenario = replace(
                        base,
                        required_capabilities=required,
                        input_type=input_type,
                        permission_values=permission_values,
                    )
                    if classify_scenario(scenario) != case_kind:
                        continue
                    if case_kind == "multiple_missing":
                        answer = solve(scenario)
                        if multiple_outcome == "tool" and answer in {
                            NO_ELIGIBLE_TOOL,
                            INSUFFICIENT_INFORMATION,
                        }:
                            continue
                        if multiple_outcome == "none" and answer != NO_ELIGIBLE_TOOL:
                            continue
                        if multiple_outcome == "conflict" and answer != INSUFFICIENT_INFORMATION:
                            continue
                    yield scenario


def _generate_row(
    config: ToolGenerationConfig,
    split: SplitName,
    index: int,
    tool_count: int,
    case_kind: str,
    multiple_outcome: str,
    dataset_id: str,
    seen_catalogs: set[str],
) -> ToolDatum:
    vocabulary = VOCABULARIES[split]
    for attempt in range(_MAX_CATALOG_ATTEMPTS):
        rng = _rng(config, split, index, str(attempt))
        base = _base_scenario(vocabulary, rng, tool_count)
        structure = structural_catalog_hash(base)
        if split_for_structure(structure) != split or structure in seen_catalogs:
            continue
        queries = tuple(_query_candidates(base, case_kind, multiple_outcome))
        if not queries:
            continue
        scenario = rng.choice(queries)
        if classify_scenario(scenario) != case_kind:
            raise AssertionError("selected query does not match its requested case kind")

        menu_rng = _rng(config, split, index, "menu")
        option_order = list(
            (*tuple(tool.id for tool in scenario.tools), NO_ELIGIBLE_TOOL, INSUFFICIENT_INFORMATION)
        )
        menu_rng.shuffle(option_order)
        template_id = TEMPLATE_IDS[split]
        request = render_tool_request(
            scenario, option_order=tuple(option_order), template_id=template_id
        )
        verify_tool_request(request, scenario)
        parsed = parse_tool_prompt(render_prompt(request))
        if parsed.scenario != scenario or parsed.option_ids != tuple(option_order):
            raise AssertionError("rendered tool prompt did not round-trip exactly")

        answer_id = solve(scenario)
        if answer_id not in option_order:
            raise AssertionError("solver answer is missing from the complete menu")
        identity_payload = json.dumps(
            (VERSION, config.seed, split, index), separators=(",", ":")
        ).encode("utf-8")
        record_id = "tool-record-" + hashlib.sha256(identity_payload).hexdigest()
        record = DecisionRecord(
            record_id=record_id,
            dataset_id=dataset_id,
            source_group_id="tool-group-" + structure,
            request=request,
            answer_id=answer_id,
        )
        seen_catalogs.add(structure)
        return ToolDatum(split, scenario, record, template_id, case_kind)

    raise ValueError(
        f"could not build {split} row {index} after {_MAX_CATALOG_ATTEMPTS} catalog attempts"
    )
