import builtins
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from reflex_decisions import baseline_data
from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import DecisionRequest, Option


def load_launcher():
    path = Path(__file__).parents[1] / "experiments" / "modal_baseline.py"
    spec = importlib.util.spec_from_file_location("modal_baseline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_prepared_inputs(directory: Path, *, record_count: int):
    records_path = directory / "records.jsonl"
    manifest_path = directory / "manifest.json"
    records = tuple(
        DecisionRecord(
            record_id=f"{baseline_data.DATASET_ID}-{index}",
            dataset_id=baseline_data.DATASET_ID,
            source_group_id=str(index),
            request=DecisionRequest(
                context=f"Premise {index}.",
                question=(
                    "Which alternative is the more plausible cause?"
                    if index % 2
                    else "Which alternative is the more plausible effect?"
                ),
                options=(
                    Option(id="choice1", label=f"First answer {index}."),
                    Option(id="choice2", label=f"Second answer {index}."),
                ),
            ),
            answer_id="choice1",
        )
        for index in range(record_count)
    )
    records_path.write_bytes(baseline_data.serialize_records(records))
    manifest = {
        "data_kind": "benchmark",
        "datasets": [
            {
                "dataset_id": baseline_data.DATASET_ID,
                "source_id": baseline_data.SOURCE_ID,
                "task_family": baseline_data.TASK_FAMILY,
                "split": "development",
                "source_uri": baseline_data.SOURCE_ARCHIVE_URI,
                "source_revision": baseline_data.SOURCE_REVISION,
                "license": baseline_data.SOURCE_LICENSE,
            }
        ],
        "held_out_families": [],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return records_path, manifest_path


def test_default_launcher_emits_bounded_plan_without_modal_or_torch(monkeypatch, capsys):
    launcher = load_launcher()
    original_import = builtins.__import__

    def forbid_remote_import(name, *args, **kwargs):
        if name == "modal" or name.startswith("torch"):
            raise AssertionError(f"unexpected local remote dependency import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_remote_import)

    assert launcher.main([]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "plan-only"
    assert plan["models"] == ["qwen", "intern", "kev"]
    assert plan["records"] == 32
    assert plan["presentations_per_model"] == 64
    assert plan["modal"]["max_containers"] == 3
    assert plan["runtime"]["torchvision"] == "0.29.1+cu130"


def test_qwen_adapter_rejects_any_nonempty_public_loading_diagnostic():
    path = Path(__file__).parents[1] / "experiments" / "baseline_qwen.py"
    spec = importlib.util.spec_from_file_location("baseline_qwen", path)
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    assert adapter.AUXILIARY_FORWARD_COUNT == 0

    with pytest.raises(RuntimeError, match="loading diagnostics are not clean"):
        adapter.validate_loading_info(
            {
                "missing_keys": [],
                "unexpected_keys": ["model.visual.weight"],
                "mismatched_keys": [],
                "error_msgs": [],
            }
        )


def test_remote_payload_omits_gold_answer_and_keeps_reversal_semantic_ids():
    launcher = load_launcher()
    record = DecisionRecord(
        record_id="copa-dev-pilot-v1-42",
        dataset_id="copa-dev-pilot-v1",
        source_group_id="42",
        request=DecisionRequest(
            context="The vase broke.",
            question="Which alternative is the more plausible cause?",
            options=(
                Option(id="choice1", label="The table shook."),
                Option(id="choice2", label="It fell."),
            ),
        ),
        answer_id="choice2",
    )

    payload = launcher._serialize_remote_records((record,))
    presentations = launcher._presentations(payload)

    assert set(payload[0]) == {"record_id", "request"}
    assert "answer_id" not in payload[0]
    assert [row["option_ids"] for row in presentations] == [
        ["choice1", "choice2"],
        ["choice2", "choice1"],
    ]
    assert presentations[1]["request"]["options"][0]["id"] == "choice2"


def test_source_fingerprints_cover_every_packaged_python_file():
    launcher = load_launcher()
    root = Path(__file__).parents[1]
    expected_paths = {
        str(path.relative_to(root))
        for package in (root / "experiments", root / "src" / "reflex_decisions")
        for path in package.rglob("*.py")
    }

    fingerprints = launcher._source_fingerprints()

    assert set(fingerprints) == expected_paths
    assert all(len(digest) == 64 for digest in fingerprints.values())


def test_initial_result_records_baseline_protocol_sha256():
    launcher = load_launcher()
    protocol_path = Path(launcher.__file__).parents[1] / launcher.EXPECTED_PROTOCOL

    result = launcher._initial_result(None, None)

    assert (
        result["provenance"]["baseline_protocol_sha256"]
        == hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    )


def test_partial_model_failure_cannot_mark_comparison_passed():
    launcher = load_launcher()
    record_ids = [f"copa-dev-pilot-v1-{index}" for index in range(32)]
    presentations = [
        {"record_id": record_id, "permutation_index": permutation}
        for record_id in record_ids
        for permutation in (0, 1)
    ]
    auxiliary_counts = {"qwen": 0, "intern": 1, "kev": 2}
    receipts = {
        name: {
            "status": "passed",
            "presentations": presentations,
            "scored_presentation_count": 64,
            "auxiliary_forward_count": auxiliary_counts[name],
            "total_forward_count": 64 + auxiliary_counts[name],
        }
        for name in auxiliary_counts
    }
    assert launcher._comparison_passed(receipts, record_ids) is True
    receipts["intern"] = {
        "status": "failed",
        "presentations": presentations[:11],
    }

    assert launcher._comparison_passed(receipts, record_ids) is False


def test_modal_app_lifecycle_failure_preserves_receipts_but_fails_run(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    launcher = load_launcher()
    records_path, manifest_path = write_prepared_inputs(tmp_path, record_count=32)
    records = tuple(
        DecisionRecord.model_validate_json(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    )

    class FakeImage:
        @classmethod
        def debian_slim(cls, **kwargs):
            return cls()

        def env(self, *_args, **_kwargs):
            return self

        def pip_install(self, *_args, **_kwargs):
            return self

        def add_local_python_source(self, *_args, **_kwargs):
            return self

    class FakeRunModel:
        def map(self, payloads, *, return_exceptions):
            assert return_exceptions is True
            return [
                {
                    "model_name": payload["model_name"],
                    "status": "passed",
                    "scored_presentation_count": 64,
                    "auxiliary_forward_count": {
                        "qwen": 0,
                        "intern": 1,
                        "kev": 2,
                    }[payload["model_name"]],
                    "total_forward_count": {
                        "qwen": 64,
                        "intern": 65,
                        "kev": 66,
                    }[payload["model_name"]],
                    "presentations": [
                        {
                            "record_id": row["record_id"],
                            "permutation_index": row["permutation_index"],
                        }
                        for row in launcher._presentations(payload["records"])
                    ],
                }
                for payload in payloads
            ]

    class FakeApp:
        def __init__(self, *_args, **_kwargs):
            pass

        def function(self, **_kwargs):
            return lambda _function: FakeRunModel()

        def run(self):
            class ExitFailure:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    raise RuntimeError("simulated Modal app shutdown failure")

            return ExitFailure()

    modal = SimpleNamespace(
        Image=FakeImage,
        App=FakeApp,
        enable_output=nullcontext,
    )
    monkeypatch.setattr(launcher.importlib, "import_module", lambda _name: modal)
    monkeypatch.setattr(launcher.importlib.metadata, "version", lambda _name: "1.6.1")
    monkeypatch.setattr(launcher.modal_smoke, "_verify_profile", lambda *_args: True)
    for name in launcher.modal_smoke.CREDENTIAL_OVERRIDES:
        monkeypatch.delenv(name, raising=False)
    output = tmp_path / "run.json"

    with launcher.smoke.reserve_output(str(output)) as reservation:
        status = launcher._launch(
            "reflex-personal",
            "rajath-61258",
            str(records_path),
            str(manifest_path),
            records,
            reservation,
        )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert status != 0
    assert result["status"] == "failed"
    assert result["failure"]["stage"] == "modal_lifecycle"
    assert all(receipt["status"] == "passed" for receipt in result["models"].values())


def test_remote_worker_rejects_unknown_model_without_loading_runtime(monkeypatch):
    launcher = load_launcher()
    original_import = builtins.__import__

    def forbid_runtime(name, *args, **kwargs):
        if name == "torch" or name.startswith("transformers"):
            raise AssertionError(f"unknown model reached runtime import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_runtime)
    receipt = launcher._remote_model_run({"model_name": "unknown", "records": []})

    assert receipt["status"] == "failed"
    assert receipt["failure"]["type"] == "UnknownModel"


@pytest.mark.parametrize(
    ("model_name", "model_id", "temperature", "auxiliary_count"),
    [
        ("qwen", "Qwen/Qwen3.5-0.8B-Base", 1.0, 0),
        ("intern", "internlm/Intern-Decision-0.8B", 2.747760550703, 1),
        ("kev", "jaredpalmer/kev-0.8b", 2.3511, 2),
    ],
)
@pytest.mark.parametrize("fail_on_call", [None, 5])
def test_remote_worker_records_forward_counts_only_after_complete_scoring(
    tmp_path, monkeypatch, model_name, model_id, temperature, auxiliary_count, fail_on_call
):
    from contextlib import nullcontext
    from types import SimpleNamespace

    launcher = load_launcher()
    records_path, _ = write_prepared_inputs(tmp_path, record_count=32)
    records = tuple(
        DecisionRecord.model_validate_json(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    )
    provenance = {
        "model_id": model_id,
        "model_revision": "fixture-revision",
        "temperature": temperature,
        "auxiliary_forward_count": 0 if model_name == "intern" else auxiliary_count,
    }
    if model_name == "intern":
        provenance["calibration_parity_passed"] = False
    score_calls = 0

    class Adapter:
        MODEL_ID = provenance["model_id"]
        MODEL_REVISION = provenance["model_revision"]
        REVISION = provenance["model_revision"]
        TEMPERATURE = provenance["temperature"]

        @staticmethod
        def load_scorer():
            def score(_request):
                nonlocal score_calls
                score_calls += 1
                if score_calls == fail_on_call:
                    raise RuntimeError("fixture scorer failure")
                if model_name == "intern":
                    provenance["calibration_parity_passed"] = True
                    provenance["auxiliary_forward_count"] = 1
                return {
                    "raw_logits": [0.2, -0.1],
                    "input_tokens": 12,
                    "prompt_sha256": "a" * 64,
                }

            return score, provenance

    cuda_calls = []

    def reset_peak_memory_stats(_device):
        cuda_calls.append("reset")
        if cuda_calls != ["init", "reset"]:
            raise RuntimeError("CUDA must be initialized before resetting peak memory stats")

    cuda = SimpleNamespace(
        is_available=lambda: True,
        init=lambda: cuda_calls.append("init"),
        reset_peak_memory_stats=reset_peak_memory_stats,
        max_memory_allocated=lambda _device: 4096,
    )
    torch = SimpleNamespace(
        cuda=cuda,
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True),
        ),
        inference_mode=nullcontext,
    )
    monkeypatch.setitem(launcher.sys.modules, "torch", torch)
    monkeypatch.setattr(launcher.importlib.util, "find_spec", lambda _name: None)
    for name in ("kernels", "fla", "causal_conv1d"):
        monkeypatch.delitem(launcher.sys.modules, name, raising=False)
    monkeypatch.setattr(
        launcher.importlib,
        "import_module",
        lambda name: Adapter if name == f"experiments.baseline_{model_name}" else pytest.fail(name),
    )
    monkeypatch.setattr(launcher.importlib.metadata, "version", lambda _name: "2.14.1")

    receipt = launcher._remote_model_run(
        {"model_name": model_name, "records": launcher._serialize_remote_records(records)}
    )

    assert cuda_calls == ["init", "reset"]
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    if fail_on_call is None:
        assert receipt["status"] == "passed"
        assert score_calls == 64
        assert receipt["scored_presentation_count"] == 64
        assert receipt["auxiliary_forward_count"] == auxiliary_count
        assert receipt["total_forward_count"] == 64 + auxiliary_count
        if model_name == "intern":
            assert receipt["provenance"]["calibration_parity_passed"] is True
    else:
        assert receipt["status"] == "failed"
        assert score_calls == fail_on_call
        assert receipt["failure"]["stage"] == "inference"
        assert (
            not {
                "scored_presentation_count",
                "auxiliary_forward_count",
                "total_forward_count",
            }
            & receipt.keys()
        )


def test_invalid_record_count_is_written_as_failure_before_auth(tmp_path, monkeypatch):
    launcher = load_launcher()
    records_path, manifest_path = write_prepared_inputs(tmp_path, record_count=31)
    monkeypatch.setattr(
        launcher.baseline_data,
        "EXPECTED_RECORDS_SHA256",
        hashlib.sha256(records_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        launcher.baseline_data,
        "EXPECTED_MANIFEST_SHA256",
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )

    def forbidden_auth(*args, **kwargs):
        raise AssertionError("auth must not run before records pass preflight")

    monkeypatch.setattr(launcher.modal_smoke, "_verify_profile", forbidden_auth)
    output = tmp_path / "failure.json"
    status = launcher.main(
        [
            "--launch",
            "--records",
            str(records_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
        ]
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert status != 0
    assert result["status"] == "failed"
    assert result["failure"]["stage"] == "input_preflight"


def test_missing_protocol_blocks_data_validation_and_auth(tmp_path, monkeypatch):
    launcher = load_launcher()
    launcher.EXPECTED_PROTOCOL = "docs/missing-baseline-protocol.md"
    records_path, manifest_path = write_prepared_inputs(tmp_path, record_count=32)
    monkeypatch.setattr(
        launcher.baseline_data,
        "verify_prepared_data",
        lambda *args: pytest.fail("protocol preflight must precede record validation"),
    )
    monkeypatch.setattr(
        launcher.modal_smoke,
        "_verify_profile",
        lambda *args: pytest.fail("protocol preflight must precede Modal auth"),
    )
    output = tmp_path / "failure.json"

    status = launcher.main(
        [
            "--launch",
            "--records",
            str(records_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
        ]
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert status != 0
    assert result["failure"]["stage"] == "protocol_preflight"
    assert result["provenance"]["baseline_protocol_sha256"] is None


@pytest.mark.parametrize("mutation", ["test_split", "fixture_manifest"])
def test_invalid_split_or_manifest_is_rejected_before_auth(tmp_path, monkeypatch, mutation):
    launcher = load_launcher()
    records_path, manifest_path = write_prepared_inputs(tmp_path, record_count=32)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "test_split":
        manifest["datasets"][0]["split"] = "test"
    else:
        manifest["data_kind"] = "fixture"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        launcher.baseline_data,
        "EXPECTED_RECORDS_SHA256",
        hashlib.sha256(records_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        launcher.baseline_data,
        "EXPECTED_MANIFEST_SHA256",
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )

    def forbidden_auth(*args, **kwargs):
        raise AssertionError("auth must not run before manifest preflight")

    monkeypatch.setattr(launcher.modal_smoke, "_verify_profile", forbidden_auth)
    output = tmp_path / "failure.json"
    status = launcher.main(
        [
            "--launch",
            "--records",
            str(records_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
        ]
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert status != 0
    assert result["failure"]["stage"] == "input_preflight"


def test_existing_output_blocks_input_validation_and_auth(tmp_path, monkeypatch):
    launcher = load_launcher()
    output = tmp_path / "existing.json"
    original = b"keep this artifact\n"
    output.write_bytes(original)

    def forbidden_preflight(*args, **kwargs):
        raise AssertionError("input validation must not run for an occupied output")

    def forbidden_auth(*args, **kwargs):
        raise AssertionError("auth must not run for an occupied output")

    monkeypatch.setattr(launcher.baseline_data, "verify_prepared_data", forbidden_preflight)
    monkeypatch.setattr(launcher.modal_smoke, "_verify_profile", forbidden_auth)
    status = launcher.main(["--launch", "--output", str(output)])

    assert status != 0
    assert output.read_bytes() == original


def test_scorer_output_requires_finite_ordered_logits_and_bounded_prompt():
    launcher = load_launcher()
    presentation = launcher._presentations(
        [
            {
                "record_id": "copa-dev-pilot-v1-42",
                "request": {
                    "context": "The vase broke.",
                    "question": "Which alternative is the more plausible cause?",
                    "options": [
                        {"id": "choice1", "label": "The table shook."},
                        {"id": "choice2", "label": "It fell."},
                    ],
                },
            }
        ]
    )[0]
    valid = {
        "raw_logits": [0.3, -0.2],
        "input_tokens": 12,
        "prompt_sha256": "a" * 64,
    }

    accepted = launcher._validated_presentation_result(presentation, valid)
    assert accepted["option_ids"] == ["choice1", "choice2"]
    assert accepted["raw_logits"] == [0.3, -0.2]
    with pytest.raises(ValueError, match="different number of logits"):
        launcher._validated_presentation_result(presentation, {**valid, "raw_logits": [0.3]})
    with pytest.raises(ValueError, match="non-finite"):
        launcher._validated_presentation_result(
            presentation, {**valid, "raw_logits": [float("nan"), -0.2]}
        )
    with pytest.raises(ValueError, match="token count"):
        launcher._validated_presentation_result(presentation, {**valid, "input_tokens": 2049})


@pytest.mark.parametrize(
    ("model_name", "temperature", "pinned_temperature", "accepted"),
    [
        ("qwen", 1.0, 1.0, True),
        ("qwen", 1.1, 1.0, False),
        ("intern", 2.747760550703, 2.747760550703, True),
        ("intern", 1.0, 2.747760550703, False),
        ("kev", 0.0, None, False),
        ("kev", float("nan"), None, False),
        ("kev", 1.25, None, True),
    ],
)
def test_adapter_provenance_requires_valid_pinned_temperature(
    model_name, temperature, pinned_temperature, accepted
):
    from types import SimpleNamespace

    launcher = load_launcher()
    adapter = SimpleNamespace(
        MODEL_ID="fixture/model",
        MODEL_REVISION="fixture-revision",
        REVISION="fixture-revision",
        TEMPERATURE=pinned_temperature,
    )
    provenance = {
        "model_id": "fixture/model",
        "model_revision": "fixture-revision",
        "temperature": temperature,
    }

    if accepted:
        assert launcher.validate_adapter_provenance(model_name, adapter, provenance) is provenance
    else:
        with pytest.raises(ValueError, match="temperature"):
            launcher.validate_adapter_provenance(model_name, adapter, provenance)
