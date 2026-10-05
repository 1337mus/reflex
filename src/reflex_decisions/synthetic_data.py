"""Reproducible synthetic seed records with exact, programmatic labels."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .data import DatasetSpec, DecisionRecord, SplitManifest, SplitName, audit_splits
from .schema import DecisionRequest, Option

DEFAULT_SEED = 20261006
DEFAULT_SCENARIO_COUNTS = (100, 25, 25)
MAX_SCENARIOS = 10_000
MAX_SCENARIOS_PER_FAMILY_SPLIT = MAX_SCENARIOS
SPLITS: tuple[SplitName, ...] = ("train", "development", "calibration")
FACT = "atomic-fact-inference"
NUMERIC = "numeric-selection"
FACT_FAMILY, NUMERIC_FAMILY = FACT, NUMERIC
RELATIONS = ("entailed", "contradicted", "unknown")
FACT_RELATIONS = RELATIONS
OPTION_COUNTS = (2, 4, 8, 16)
VERSION = "synthetic-seed-data-v1"
GENERATOR_VERSION = VERSION
SOURCE = {FACT: "synthetic-atomic-fact-inference-v1", NUMERIC: "synthetic-numeric-selection-v1"}
BANKS = {
    FACT: {s: (f"fact-{s}-complete-a-v1", f"fact-{s}-complete-b-v1") for s in SPLITS},
    NUMERIC: {s: (f"numeric-{s}-values-a-v1", f"numeric-{s}-values-b-v1") for s in SPLITS},
}
FACT_TEMPLATE_BANKS, NUMERIC_TEMPLATE_BANKS = BANKS[FACT], BANKS[NUMERIC]
WORDING = {
    FACT: {
        "train": (
            (
                "Rule: each listed object has exactly one color; this is the full list.",
                "From these facts, mark the exact claim entailed, contradicted, or unknown:",
            ),
            (
                "All available color facts appear below; every object has one color.",
                "Do these facts entail, contradict, or leave open this exact color claim:",
            ),
        ),
        "development": (
            (
                "The complete color record is shown; each listed object has one color.",
                "Does the full record entail, contradict, or leave this claim unknown?",
            ),
            (
                "For each listed object, its sole color is shown; no other facts are available.",
                "Is this claim entailed, contradicted, or unknown from the full record?",
            ),
        ),
        "calibration": (
            (
                "Treat the following as the entire record; each listed object has one color.",
                "Choose for this exact statement: entailed, contradicted, or unknown?",
            ),
            (
                "This is the complete set of color facts; each object has exactly one color.",
                "Do these facts establish, rule out, or leave open the exact claim?",
            ),
        ),
    },
    NUMERIC: {
        "train": (
            (
                "Exact measurements for every candidate item:",
                "Which item has the {direction} measurement?",
            ),
            (
                "Candidate values are recorded exactly below:",
                "Name the item with the {direction} value.",
            ),
        ),
        "development": (
            (
                "The following table gives each named item's exact value:",
                "Select the item whose measured value is {direction}.",
            ),
            (
                "Use these complete, exact measurements for the listed items:",
                "Which listed item has the {direction} value in this table?",
            ),
        ),
        "calibration": (
            (
                "Each candidate's measurement is exact and shown below:",
                "Choose the candidate with the {direction} recorded measurement.",
            ),
            (
                "Here are the exact values for all available candidates:",
                "Identify the item whose value is {direction} among these candidates.",
            ),
        ),
    },
}
COLORS = tuple(
    "amber blue coral green indigo ochre violet white yellow silver teal crimson "
    "cyan gold magenta navy".split()
)
SYLLABLES = tuple(
    "ba be bi bo bu ca ce ci co cu da de di do du fa fe fi fo fu ga ge gi go gu "
    "la le li lo lu ma".split()
)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    seed: int = DEFAULT_SEED
    fact_scenarios: tuple[int, int, int] = DEFAULT_SCENARIO_COUNTS
    numeric_scenarios: tuple[int, int, int] = DEFAULT_SCENARIO_COUNTS

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for counts in (self.fact_scenarios, self.numeric_scenarios):
            if len(counts) != len(SPLITS) or any(
                type(n) is not int or not 1 <= n <= MAX_SCENARIOS for n in counts
            ):
                raise ValueError(f"each split needs 1–{MAX_SCENARIOS} scenarios per family")


@dataclass(frozen=True, slots=True)
class SyntheticCandidate:
    config: GenerationConfig
    manifest: SplitManifest
    records: tuple[DecisionRecord, ...]
    provenance: tuple[dict[str, Any], ...]
    recipe: dict[str, Any]
    audit: dict[str, Any]


def _json(value: object, *, pretty: bool = False) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )
        + "\n"
    ).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_json(value).rstrip(b"\n")).hexdigest()


def _id(prefix: str, seed: int, *parts: object) -> str:
    return f"{prefix}_{_digest([VERSION, seed, *parts])[:24]}"


def _name(scenario: int, slot: int) -> str:
    value = (scenario * 16 + slot) % len(SYLLABLES) ** 4
    return "".join(
        reversed([SYLLABLES[(value // len(SYLLABLES) ** i) % len(SYLLABLES)] for i in range(4)])
    )


def solve_atomic_fact(facts: Mapping[str, str], subject: str, color: str) -> str:
    known = facts.get(subject)
    return "unknown" if known is None else "entailed" if known == color else "contradicted"


def solve_numeric_selection(
    measurements: Mapping[str, int], direction: Literal["smallest", "largest"]
) -> str:
    if direction not in {"smallest", "largest"} or not measurements:
        raise ValueError("direction must be smallest/largest and measurements nonempty")
    values = tuple(measurements.values())
    if any(type(value) is not int for value in values) or len(values) != len(set(values)):
        raise ValueError("measurements must be unique integers")
    chooser = min if direction == "smallest" else max
    return chooser(measurements, key=measurements.__getitem__)


def validate_scenario_hash_partition(provenance: Iterable[Mapping[str, Any]]) -> None:
    seen: dict[str, str] = {}
    for row in provenance:
        digest, split = row.get("canonical_problem_sha256"), row.get("split")
        if not isinstance(digest, str) or not isinstance(split, str):
            raise ValueError("scenario provenance is missing a problem hash or split")
        if digest in seen:
            if seen[digest] != split:
                raise ValueError("a canonical scenario problem appears across data splits")
            raise ValueError("duplicate canonical scenario problem within a split")
        seen[digest] = split


def _stable_int(seed: int, scenario: int, counter: int) -> int:
    return (
        int(_digest([VERSION, seed, NUMERIC, scenario, "measurement", counter])[:16], 16) % 20_001
        - 10_000
    )


def _scenario(
    config: GenerationConfig, family: str, split: SplitName, index: int, local: int
) -> tuple[list[DecisionRecord], dict[str, Any]]:
    is_fact = family == FACT
    variant = local % 2 if is_fact else (local // len(OPTION_COUNTS)) % 2
    template = BANKS[family][split][variant]
    context_intro, question_intro = WORDING[family][split][variant]
    scenario_id = _id("scenario", config.seed, family, index)
    group_id = _id("grp", config.seed, family, index)
    dataset_id = f"synthetic-{family}-v1-{split}"

    if is_fact:
        colors = sorted(COLORS, key=lambda c: _digest([VERSION, config.seed, family, index, c]))
        names = [_name(index, slot) for slot in range(4)]
        facts = {names[slot]: colors[slot] for slot in range(3)}
        context = context_intro + "\nAn unlisted object's color is unknown.\nFacts:\n"
        context += "\n".join(f"- {n} has exactly color {c}." for n, c in sorted(facts.items()))
        known_color = facts[names[0]]
        other_color = min(
            (color for color in COLORS if color != known_color),
            key=lambda color: _digest(
                [VERSION, config.seed, family, index, "contradiction-color", color]
            ),
        )
        claims = ((names[0], known_color), (names[0], other_color), (names[3], colors[3]))
        options = tuple(
            Option(
                id=_id("opt", config.seed, family, index, r),
                label=r,
                description=f"The stated facts {d} the claim.",
            )
            for r, d in zip(RELATIONS, ("establish", "rule out", "do not determine"), strict=True)
        )
        queries = [
            (
                f"{question_intro} “{n} has color {c}.”",
                solve_atomic_fact(facts, n, c),
                {"subject": n, "color": c},
            )
            for n, c in claims
        ]
        problem: dict[str, Any] = {
            "facts": [{"subject": n, "color": c} for n, c in sorted(facts.items())],
            "claims": sorted(
                [{"subject": n, "color": c} for n, c in claims],
                key=lambda x: (x["subject"], x["color"]),
            ),
        }
    else:
        count = OPTION_COUNTS[local % len(OPTION_COUNTS)]
        item_names = tuple(_name(index, slot) for slot in range(count))
        values: list[int] = [-1, 0] if index == 0 and count >= 2 else []
        counter = 0
        while len(values) < count:
            value = _stable_int(config.seed, index, counter)
            counter += 1
            if value not in values:
                values.append(value)
        measurements = dict(zip(item_names, values, strict=True))
        context = context_intro + "\n"
        context += "\n".join(f"- {n} has value {measurements[n]}." for n in sorted(item_names))
        queries = [
            (
                question_intro.format(direction=d),
                solve_numeric_selection(measurements, d),
                {"direction": d},
            )
            for d in ("smallest", "largest")
        ]
        problem = {
            "measurements": [{"item": n, "value": measurements[n]} for n in sorted(item_names)],
            "queries": ["smallest", "largest"],
        }
        options = tuple(
            Option(
                id=_id("opt", config.seed, family, index, n),
                label=n,
                description="One of the named items in the exact measurement list.",
            )
            for n in sorted(
                item_names,
                key=lambda n: _digest([VERSION, config.seed, family, index, "option-order", n]),
            )
        )

    if is_fact:
        options = tuple(
            sorted(
                options,
                key=lambda o: _digest([VERSION, config.seed, family, index, "option-order", o.id]),
            )
        )
    option_id = {o.label: o.id for o in options}
    records, evidence = [], []
    for query_index, (question, answer, detail) in enumerate(queries):
        record_id = _id("rec", config.seed, family, index, query_index)
        records.append(
            DecisionRecord(
                record_id=record_id,
                dataset_id=dataset_id,
                source_group_id=group_id,
                request=DecisionRequest(context=context, question=question, options=options),
                answer_id=option_id[answer],
            )
        )
        evidence.append({"record_id": record_id, **detail, "solver_answer": answer})
    trace: dict[str, Any] = {
        "generator_version": VERSION,
        "family": family,
        "split": split,
        "scenario_id": scenario_id,
        "source_group_id": group_id,
        "canonical_problem_sha256": _digest({"family": family, "problem": problem}),
        "template_id": template,
        "structured_problem": problem,
        "derived_examples": evidence,
    }
    if not is_fact:
        trace["option_count"] = len(options)
    return records, trace


def _recipe(config: GenerationConfig) -> dict[str, Any]:
    return {
        "recipe_id": VERSION,
        "seed": config.seed,
        "data_kind": "fixture",
        "scenario_counts_by_split": {
            FACT: dict(zip(SPLITS, config.fact_scenarios, strict=True)),
            NUMERIC: dict(zip(SPLITS, config.numeric_scenarios, strict=True)),
        },
        "task_families": [FACT, NUMERIC],
        "numeric_option_count_cycle": list(OPTION_COUNTS),
        "scenario_index_policy": "split index * max scenarios per split + local scenario index",
        "max_scenarios_per_split": MAX_SCENARIOS_PER_FAMILY_SPLIT,
        "wording_template_banks": {
            f: {s: list(BANKS[f][s]) for s in SPLITS} for f in (FACT, NUMERIC)
        },
        "source_group_policy": "one scenario per source group; its questions stay together",
        "scenario_hash_policy": "canonical problems exclude split, template, and opaque IDs",
        "solver_version": VERSION,
    }


def _audit(
    manifest: SplitManifest, records: tuple[DecisionRecord, ...], traces: tuple[dict[str, Any], ...]
) -> dict[str, Any]:
    split_audit = audit_splits(manifest, records).model_dump(mode="json")
    datasets = {d.dataset_id: d for d in manifest.datasets}
    family_counts = Counter(datasets[r.dataset_id].task_family for r in records)
    split_counts = Counter(datasets[r.dataset_id].split for r in records)
    by_id = {r.record_id: r for r in records}
    scenarios: dict[str, Counter[str]] = {f: Counter() for f in (FACT, NUMERIC)}
    relations: dict[str, Counter[str]] = {f: Counter() for f in (FACT, NUMERIC)}
    positions: dict[str, Counter[int]] = {f: Counter() for f in (FACT, NUMERIC)}
    values: list[int] = []
    for trace in traces:
        family, split = trace["family"], trace["split"]
        scenarios[family][split] += 1
        for example in trace["derived_examples"]:
            record = by_id[example["record_id"]]
            key = example.get("solver_answer") if family == FACT else example["direction"]
            relations[family][str(key)] += 1
            positions[family][
                next(i for i, o in enumerate(record.request.options, 1) if o.id == record.answer_id)
            ] += 1
        if family == NUMERIC:
            values.extend(m["value"] for m in trace["structured_problem"]["measurements"])
    hashes = [t["canonical_problem_sha256"] for t in traces]
    return {
        "split_audit": split_audit,
        "record_counts_by_family": dict(sorted(family_counts.items())),
        "record_counts_by_split": {s: split_counts[s] for s in SPLITS},
        "scenario_counts_by_family_and_split": {
            f: {s: scenarios[f][s] for s in SPLITS} for f in (FACT, NUMERIC)
        },
        "relation_counts": {f: dict(sorted(relations[f].items())) for f in (FACT, NUMERIC)},
        "answer_position_counts_1_based": {
            f: {str(i): positions[f][i] for i in sorted(positions[f])} for f in (FACT, NUMERIC)
        },
        "numeric_candidate_values": {
            "negative_count": sum(v < 0 for v in values),
            "zero_count": values.count(0),
            "positive_count": sum(v > 0 for v in values),
        },
        "scenario_hashes": {
            "count": len(hashes),
            "unique_count": len(set(hashes)),
            "cross_split_duplicates": 0,
        },
    }


def build_candidate(config: GenerationConfig | None = None) -> SyntheticCandidate:
    config = config or GenerationConfig()
    datasets: list[DatasetSpec] = []
    records: list[DecisionRecord] = []
    traces: list[dict[str, Any]] = []
    settings = (
        (FACT, config.fact_scenarios, "atomic-fact-inference"),
        (NUMERIC, config.numeric_scenarios, "numeric-selection"),
    )
    for family, counts, task in settings:
        datasets.extend(
            DatasetSpec(
                dataset_id=f"synthetic-{family}-v1-{split}",
                source_id=SOURCE[family],
                task_family=task,
                split=split,
                source_uri=f"synthetic://reflex/{family}/v1/{split}",
                source_revision=VERSION,
                license="self-authored synthetic",
            )
            for split in SPLITS
        )
        for split_index, (split, count) in enumerate(zip(SPLITS, counts, strict=True)):
            for local in range(count):
                index = split_index * MAX_SCENARIOS_PER_FAMILY_SPLIT + local
                made, trace = _scenario(config, family, split, index, local)
                records.extend(made)
                traces.append(trace)
    provenance, record_tuple = tuple(traces), tuple(records)
    validate_scenario_hash_partition(provenance)
    if len({r.record_id for r in record_tuple}) != len(record_tuple):
        raise ValueError("generated record IDs are not unique")
    manifest = SplitManifest(
        data_kind="fixture",
        datasets=tuple(datasets),
        group_partitioned_sources=tuple(SOURCE.values()),
    )
    return SyntheticCandidate(
        config,
        manifest,
        record_tuple,
        provenance,
        _recipe(config),
        _audit(manifest, record_tuple, provenance),
    )


def _jsonl(rows: Iterable[Any]) -> bytes:
    return b"".join(_json(row).rstrip(b"\n") + b"\n" for row in rows)


def write_candidate(candidate: SyntheticCandidate, output_dir: str | Path) -> dict[str, Any]:
    """Write output to a new directory and return hashes for each data file."""
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("output directory already exists; choose a fresh directory")
    files = {
        "records.jsonl": _jsonl(r.model_dump(mode="json") for r in candidate.records),
        "manifest.json": _json(candidate.manifest.model_dump(mode="json"), pretty=True),
        "recipe.json": _json(candidate.recipe, pretty=True),
        "provenance.jsonl": _jsonl(candidate.provenance),
        "audit.json": _json(candidate.audit, pretty=True),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    created = []
    try:
        for name, content in files.items():
            path = target / name
            with path.open("xb") as handle:
                created.append(path)
                handle.write(content)
        report = {
            "schema_version": 1,
            "generator_version": VERSION,
            "seed": candidate.config.seed,
            "counts": candidate.audit,
            "file_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        }
        report_path = target / "report.json"
        with report_path.open("xb") as handle:
            created.append(report_path)
            handle.write(_json(report, pretty=True))
        return report
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        target.rmdir()
        raise
