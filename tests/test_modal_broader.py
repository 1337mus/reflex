import builtins
import importlib.util
import json
from pathlib import Path


def load_launcher():
    path = Path(__file__).parents[1] / "experiments" / "modal_broader.py"
    spec = importlib.util.spec_from_file_location("modal_broader", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_launcher_emits_bounded_plan_without_remote_imports(monkeypatch, capsys):
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
    assert plan["records"] == 64
    assert plan["presentations_per_model"] == 256
    assert plan["presentations_total"] == 768
    assert plan["presentations_per_model_by_dataset"] == {
        "boolq-dev-pilot-v1": 64,
        "snli-dev-pilot-v1": 192,
    }
    assert plan["auxiliary_forward_counts"] == {"qwen": 0, "intern": 1, "kev": 2}
    assert plan["total_forward_count_per_comparison"] == 771
    assert plan["modal"]["gpu"] == "A10"
    assert plan["modal"]["max_containers"] == 3
    assert plan["modal"]["retries"] == 0


def _records():
    from reflex_decisions.data import DecisionRecord
    from reflex_decisions.schema import DecisionRequest, Option

    rows = [
        DecisionRecord(
            record_id=f"boolq-dev-pilot-v1-{index}",
            dataset_id="boolq-dev-pilot-v1",
            source_group_id=f"boolq-group-{index}",
            request=DecisionRequest(
                context=f"Passage {index}.",
                question=f"Question {index}?",
                options=(Option(id="yes", label="Yes"), Option(id="no", label="No")),
            ),
            answer_id="yes",
        )
        for index in range(32)
    ]
    rows += [
        DecisionRecord(
            record_id=f"snli-dev-pilot-v1-{index}",
            dataset_id="snli-dev-pilot-v1",
            source_group_id=f"snli-group-{index}",
            request=DecisionRequest(
                context=f"Premise {index}.",
                question=f"Hypothesis {index}.",
                options=(
                    Option(id="entailment", label="Entailment"),
                    Option(id="neutral", label="Neutral"),
                    Option(id="contradiction", label="Contradiction"),
                ),
            ),
            answer_id="entailment",
        )
        for index in range(32)
    ]
    return tuple(rows)


def test_modal_lifecycle_failure_preserves_complete_model_receipts(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from reflex_decisions.smoke import reserve_output

    launcher = load_launcher()
    records = _records()
    records_path = tmp_path / "records.jsonl"
    manifest_path = tmp_path / "manifest.json"
    records_path.write_text("synthetic fixture\n", encoding="utf-8")
    manifest_path.write_text("synthetic fixture\n", encoding="utf-8")

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
                    "provenance": {},
                    "presentations": [
                        {
                            "record_id": row["record_id"],
                            "request_hash": row["request_hash"],
                            "permutation_index": row["permutation_index"],
                            "option_ids": row["option_ids"],
                            "raw_logits": [0.0] * len(row["option_ids"]),
                            "input_tokens": 5,
                            "prompt_sha256": "a" * 64,
                        }
                        for row in launcher._presentations(payload["records"])
                    ],
                    "scored_presentation_count": 256,
                    "auxiliary_forward_count": {"qwen": 0, "intern": 1, "kev": 2}[
                        payload["model_name"]
                    ],
                    "total_forward_count": 256
                    + {"qwen": 0, "intern": 1, "kev": 2}[payload["model_name"]],
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

    fake_modal = SimpleNamespace(Image=FakeImage, App=FakeApp, enable_output=nullcontext)
    monkeypatch.setattr(launcher.modal_smoke, "_verify_profile", lambda *_args: True)
    monkeypatch.setattr(launcher.importlib.metadata, "version", lambda _name: "1.6.1")
    monkeypatch.setattr(launcher.importlib, "import_module", lambda _name: fake_modal)

    with reserve_output(tmp_path / "run.json") as reservation:
        status = launcher._launch(
            "profile",
            "workspace",
            str(records_path),
            str(manifest_path),
            records,
            reservation,
        )

    result = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert status == 1
    assert result["status"] == "failed"
    assert result["failure"]["stage"] == "modal_lifecycle"
    assert all(len(receipt["presentations"]) == 256 for receipt in result["models"].values())


def test_invalid_prepared_data_is_saved_before_auth_or_modal_import(tmp_path, monkeypatch, capsys):
    launcher = load_launcher()
    records_path = tmp_path / "records.jsonl"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "failure.json"
    records_path.write_text("synthetic input\n", encoding="utf-8")
    manifest_path.write_text("synthetic manifest\n", encoding="utf-8")

    def invalid_data(*_args):
        raise ValueError("fixture records failed preflight")

    def forbidden_auth(*_args):
        raise AssertionError("auth must not run before the exact data preflight")

    monkeypatch.setattr(launcher.broader_data, "verify_prepared_data", invalid_data, raising=False)
    monkeypatch.setattr(launcher.modal_smoke, "_verify_profile", forbidden_auth)

    status = launcher.main(
        [
            "--launch",
            "--records",
            str(records_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
        ]
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert status == 1
    assert result["failure"]["stage"] == "input_preflight"
    assert result["status"] == "failed"
    assert capsys.readouterr().out == ""
