"""Build and validate fixed role-scoped runtime-rule worker payloads."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, cast

from experiments import (
    runtime_rule_study_bundle as bundle_contract,
)
from experiments import (
    runtime_rule_study_contracts as contracts,
)
from experiments import (
    runtime_rule_study_payload_metadata as metadata,
)
from experiments import (
    runtime_rule_study_payload_rows as rows,
)
from experiments.mixture_training_contracts import (
    canonical_json,
    json_sha256,
    normalize_json_object,
    validate_sha256,
)

_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
_RUN_SUFFIXES = {
    contracts.ROLE_UNCHANGED: "-unchanged",
    contracts.ROLES[1]: "-control",
    contracts.ROLES[2]: "-runtime",
}
_COMPILATION_FIELDS = {
    "training",
    "evaluation",
    "unchanged",
    "reload",
    "schedule_sha256",
    "counts",
    "input_token_counts",
    "max_input_tokens",
    "compilation_sha256",
}
_PAYLOAD_FIELDS = {
    "schema_version",
    "experiment_id",
    "study_id",
    "run_id",
    "nonce",
    "role",
    "source_commit",
    "bundle_sha256",
    "study",
    "source_file_sha256",
    "training_pools",
    "training_specs",
    "evaluation",
    "evaluation_specs",
    "reload",
    "reload_specs",
    "optimizer",
    "worker_settings",
    "forward_counts",
    "input_token_counts",
    "payload_sha256",
}


def _validate_role(role: object) -> str:
    if not isinstance(role, str) or role not in contracts.ROLES:
        raise ValueError("role must be unchanged, continued_practice, or runtime_mix")
    return role


def _run_id(study_id: object, role: str) -> str:
    checked_study_id = contracts.validate_safe_run_id(study_id, "study_id")
    return contracts.validate_safe_run_id(f"{checked_study_id}{_RUN_SUFFIXES[role]}", "run_id")


def _validate_compilation(value: object) -> dict[str, Any]:
    compilation = normalize_json_object(value, "compilation")
    if set(compilation) != _COMPILATION_FIELDS:
        raise ValueError("compilation has an unexpected schema")
    digest = validate_sha256(compilation["compilation_sha256"], "compilation SHA-256")
    unsigned = {key: item for key, item in compilation.items() if key != "compilation_sha256"}
    if json_sha256(unsigned) != digest:
        raise ValueError("compilation SHA-256 does not match its canonical fields")
    training = compilation["training"]
    if not isinstance(training, Mapping) or set(training) != set(contracts.ROLES[1:]):
        raise ValueError("compilation training roles differ from the fixed study arms")
    for role in contracts.ROLES:
        contracts.expected_token_counts(role, compilation)

    groups: dict[str, object] = {
        "training_continued_practice": training[contracts.ROLES[1]],
        "training_runtime_mix": training[contracts.ROLES[2]],
        "evaluation_both_arms": compilation["evaluation"],
        "unchanged_adapter": compilation["unchanged"],
        "reload_both_arms": compilation["reload"],
    }
    multipliers = {
        "training_continued_practice": 1,
        "training_runtime_mix": 1,
        "evaluation_both_arms": 2,
        "unchanged_adapter": 1,
        "reload_both_arms": 2,
    }
    token_counts: dict[str, int] = {}
    maximum = 0
    for name, specs_value in groups.items():
        if not isinstance(specs_value, list):
            raise ValueError(f"compiled {name} specs must be an array")
        category_tokens = 0
        for spec in specs_value:
            if not isinstance(spec, Mapping):
                raise ValueError(f"compiled {name} spec must be an object")
            input_count = spec.get("input_tokens")
            if type(input_count) is not int or not 1 <= input_count <= contracts.MAX_INPUT_TOKENS:
                raise ValueError(f"compiled {name} input token count is outside the fixed limit")
            category_tokens += input_count
            maximum = max(maximum, input_count)
        token_counts[name] = multipliers[name] * category_tokens
    token_counts["total"] = sum(token_counts.values())
    if token_counts != compilation["input_token_counts"]:
        raise ValueError("compiled input token totals differ from their individual specs")
    if compilation["max_input_tokens"] != maximum:
        raise ValueError("compiled maximum input token count differs from its individual specs")
    return compilation


def _role_rows(
    bundle: Mapping[str, object], compilation: Mapping[str, object], role: str
) -> dict[str, object]:
    if role == contracts.ROLE_UNCHANGED:
        all_evaluation = bundle.get("evaluation")
        unchanged = bundle.get("unchanged")
        all_specs = compilation.get("evaluation")
        unchanged_specs = compilation.get("unchanged")
        if (
            not isinstance(all_evaluation, list)
            or not isinstance(unchanged, list)
            or len(all_evaluation) != 3_946
            or all_evaluation[-164:] != unchanged
        ):
            raise ValueError("bundle unchanged rows differ from the fixed final panel slice")
        if (
            not isinstance(all_specs, list)
            or not isinstance(unchanged_specs, list)
            or len(all_specs) != 3_946
            or all_specs[-164:] != unchanged_specs
        ):
            raise ValueError("compiled unchanged specs differ from the fixed final panel slice")
        return {
            "training_pools": [],
            "training_specs": [],
            "evaluation": unchanged,
            "evaluation_specs": unchanged_specs,
            "reload": [],
            "reload_specs": [],
            "optimizer": None,
        }
    training = cast(Mapping[str, object], compilation["training"])
    return {
        "training_pools": bundle.get("training_pools"),
        "training_specs": training[role],
        "evaluation": bundle.get("evaluation"),
        "evaluation_specs": compilation.get("evaluation"),
        "reload": bundle.get("reload"),
        "reload_specs": compilation.get("reload"),
        "optimizer": contracts.optimizer_settings(),
    }


def build_worker_payload(
    bundle: object,
    *,
    expected_bundle_sha256: object,
    study_id: object,
    nonce: object,
    role: object,
    source_commit: object,
    source_file_sha256: object,
) -> dict[str, object]:
    """Build a detached worker payload from an authenticated CPU bundle."""

    contracts.validate_safe_run_id(study_id, "study_id")
    contracts.validate_uuid(nonce, "nonce")
    checked_role = _validate_role(role)
    run_id = _run_id(study_id, checked_role)
    expected_digest = validate_sha256(expected_bundle_sha256, "expected bundle SHA-256")
    if not isinstance(source_commit, str) or _GIT_SHA.fullmatch(source_commit) is None:
        raise ValueError("source_commit must be a full lowercase 40-character Git SHA")
    source_hashes = metadata.validate_source_file_sha256(source_file_sha256)
    authenticated = bundle_contract.verify_bundle_digest(bundle, expected_digest)
    compilation = _validate_compilation(authenticated.get("compilation"))
    study_metadata = metadata.build_study_metadata(authenticated, expected_digest, source_hashes)
    selected = _role_rows(authenticated, compilation, checked_role)
    unsigned: dict[str, object] = {
        "schema_version": contracts.SCHEMA_VERSION,
        "experiment_id": contracts.EXPERIMENT_ID,
        "study_id": study_id,
        "run_id": run_id,
        "nonce": nonce,
        "role": checked_role,
        "source_commit": source_commit,
        "bundle_sha256": expected_digest,
        "study": study_metadata,
        "source_file_sha256": source_hashes,
        **selected,
        "worker_settings": contracts.worker_settings(checked_role),
        "forward_counts": contracts.expected_forward_counts(checked_role),
        "input_token_counts": contracts.expected_token_counts(checked_role, compilation),
    }
    payload = {**unsigned, "payload_sha256": json_sha256(unsigned)}
    return validate_payload(payload)


def _validate_role_counts(payload: Mapping[str, object], role: str) -> None:
    forward_counts = payload.get("forward_counts")
    if forward_counts != contracts.expected_forward_counts(role):
        raise ValueError("payload forward counts differ from the fixed worker role")
    if not isinstance(forward_counts, Mapping) or any(
        type(count) is not int for count in forward_counts.values()
    ):
        raise ValueError("payload forward counts must be strict integers")
    tokens_by_role = payload.get("input_token_counts")
    expected_keys = {"training", "final_evaluation", "reload_parity", "total"}
    if not isinstance(tokens_by_role, Mapping) or set(tokens_by_role) != expected_keys:
        raise ValueError("payload input token counts have an unexpected schema")
    if any(type(count) is not int or count < 0 for count in tokens_by_role.values()):
        raise ValueError("payload input token counts must be strict non-negative integers")


def validate_payload(
    value: object, *, expected_payload_sha256: object | None = None
) -> dict[str, object]:
    """Validate a complete role payload without reading source or dataset files."""

    payload = normalize_json_object(value, "payload")
    if set(payload) != _PAYLOAD_FIELDS:
        raise ValueError("payload has an unexpected top-level schema")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != contracts.SCHEMA_VERSION
    ):
        raise ValueError("payload schema_version must be strict integer 1")
    if payload["experiment_id"] != contracts.EXPERIMENT_ID:
        raise ValueError("payload experiment_id is unsupported")
    contracts.validate_safe_run_id(payload["study_id"], "study_id")
    contracts.validate_uuid(payload["nonce"], "nonce")
    role = _validate_role(payload["role"])
    run_id = _run_id(payload["study_id"], role)
    if payload["run_id"] != run_id:
        raise ValueError("payload run_id differs from the study ID and fixed role suffix")
    source_commit = payload["source_commit"]
    if not isinstance(source_commit, str) or _GIT_SHA.fullmatch(source_commit) is None:
        raise ValueError("source_commit must be a full lowercase 40-character Git SHA")
    bundle_digest = validate_sha256(payload["bundle_sha256"], "bundle SHA-256")
    source_hashes = metadata.validate_source_file_sha256(payload["source_file_sha256"])
    study_metadata = metadata.validate_study_metadata(payload["study"], source_hashes)
    if study_metadata["bundle_sha256"] != bundle_digest:
        raise ValueError("payload bundle digest differs from its study metadata")
    expected_digest = validate_sha256(payload["payload_sha256"], "payload_sha256")
    if expected_payload_sha256 is not None and expected_digest != validate_sha256(
        expected_payload_sha256, "trusted expected payload SHA-256"
    ):
        raise ValueError("payload SHA-256 differs from the trusted expected digest")
    unsigned = {key: item for key, item in payload.items() if key != "payload_sha256"}
    if json_sha256(unsigned) != expected_digest:
        raise ValueError("payload SHA-256 does not match its canonical fields")

    if canonical_json(payload["worker_settings"]) != canonical_json(
        contracts.worker_settings(role)
    ):
        raise ValueError("payload worker settings differ from the frozen role contract")
    expected_optimizer = (
        None if role == contracts.ROLE_UNCHANGED else contracts.optimizer_settings()
    )
    if canonical_json(payload.get("optimizer")) != canonical_json(expected_optimizer):
        raise ValueError("payload optimizer differs from the frozen role contract")
    _validate_role_counts(payload, role)
    role_rows = rows.validate_role_rows(
        payload, role, cast(Mapping[str, str], study_metadata["schedule_sha256"])
    )
    payload.update(role_rows)
    payload["study"] = study_metadata
    payload["source_file_sha256"] = source_hashes

    forward_counts = cast(Mapping[str, int], payload["forward_counts"])
    input_counts = cast(Mapping[str, int], payload["input_token_counts"])
    expected_forward_counts = contracts.expected_forward_counts(role)
    if forward_counts != expected_forward_counts:
        raise ValueError("payload forward count map differs from the frozen role contract")
    training_tokens = sum(
        cast(int, spec["input_tokens"])
        for spec in cast(list[dict[str, object]], payload["training_specs"])
    )
    final_tokens = sum(
        cast(int, spec["input_tokens"])
        for spec in cast(list[dict[str, object]], payload["evaluation_specs"])
    )
    reload_tokens = sum(
        cast(int, spec["input_tokens"])
        for spec in cast(list[dict[str, object]], payload["reload_specs"])
    )
    derived_tokens = {
        "training": training_tokens,
        "final_evaluation": final_tokens,
        "reload_parity": reload_tokens,
        "total": training_tokens + final_tokens + reload_tokens,
    }
    if input_counts != derived_tokens:
        raise ValueError("payload input token counts differ from its compiled request rows")
    expected_train_count = 0 if role == contracts.ROLE_UNCHANGED else 1_344
    expected_reload_count = 0 if role == contracts.ROLE_UNCHANGED else 32
    if (
        forward_counts["training"] != expected_train_count
        or forward_counts["final_evaluation"] != len(payload["evaluation_specs"])
        or forward_counts["reload_parity"] != expected_reload_count
    ):
        raise ValueError("payload request row counts differ from the role forward plan")
    payload["input_token_counts"] = derived_tokens
    return payload


def training_examples(payload: object) -> tuple[object, ...]:
    """Regenerate the selected arm's deterministic training schedule."""

    normalized = validate_payload(payload)
    return rows.selected_training_examples(normalized, cast(str, normalized["role"]))
