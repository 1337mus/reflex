import importlib.util
import json
import subprocess
from pathlib import Path

from reflex_decisions import smoke


def test_default_launcher_emits_plan_without_importing_modal_or_torch(monkeypatch, capsys) -> None:
    import builtins

    modal_smoke = load_launcher()

    original_import = builtins.__import__

    def forbid_cloud_import(name, *args, **kwargs):
        if name == "modal" or name.startswith("torch"):
            raise AssertionError(f"unexpected cloud or model import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_cloud_import)
    assert modal_smoke.main([]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "plan-only"
    assert plan["model_revision"] == smoke.MODEL_REVISION


def test_output_preflight_failure_returns_before_modal_import(tmp_path, monkeypatch) -> None:
    import builtins

    modal_smoke = load_launcher()

    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("x", encoding="utf-8")

    def forbid_cloud_import(name, *args, **kwargs):
        if name == "modal" or name.startswith("torch"):
            raise AssertionError(f"unexpected cloud or model import: {name}")
        return original_import(name, *args, **kwargs)

    original_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", forbid_cloud_import)
    assert (
        modal_smoke.main(
            [
                "--launch",
                "--profile",
                "personal",
                "--workspace",
                "expected-workspace",
                "--output",
                str(parent_file / "result.json"),
            ]
        )
        != 0
    )


def test_workspace_mismatch_blocks_modal_import_and_never_prints_token_id(
    tmp_path, monkeypatch, capsys
) -> None:
    import builtins

    modal_smoke = load_launcher()

    invocation = {}

    def fake_run(command, **kwargs):
        invocation["command"] = command
        invocation["env"] = kwargs["env"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="Token ID: do-not-print-this\nWorkspace: wrong-workspace (workspace-id)\n",
            stderr="",
        )

    original_import = builtins.__import__

    def forbid_cloud_import(name, *args, **kwargs):
        if name == "modal" or name.startswith("torch"):
            raise AssertionError(f"unexpected cloud or model import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(builtins, "__import__", forbid_cloud_import)
    output = tmp_path / "result.json"
    assert (
        modal_smoke.main(
            [
                "--launch",
                "--profile",
                "personal",
                "--workspace",
                "expected-workspace",
                "--output",
                str(output),
            ]
        )
        != 0
    )

    assert invocation["command"][-2:] == ["token", "info"]
    assert invocation["env"]["MODAL_PROFILE"] == "personal"
    assert invocation["env"]["NO_COLOR"] == "1"
    assert invocation["env"]["TERM"] == "dumb"
    assert "do-not-print-this" not in capsys.readouterr().out
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["failure"]["stage"] == "auth"


def test_modal_token_environment_override_blocks_token_info(tmp_path, monkeypatch) -> None:
    modal_smoke = load_launcher()

    def forbidden_run(*args, **kwargs):
        raise AssertionError("token info must not run with a credential override")

    monkeypatch.setenv("MODAL_TOKEN_ID", "private-value")
    monkeypatch.setattr(subprocess, "run", forbidden_run)
    output = tmp_path / "result.json"
    assert (
        modal_smoke.main(
            [
                "--launch",
                "--profile",
                "personal",
                "--workspace",
                "expected-workspace",
                "--output",
                str(output),
            ]
        )
        != 0
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["failure"]["stage"] == "environment"
    assert "private-value" not in output.read_text(encoding="utf-8")


def load_launcher():
    path = Path(__file__).parents[1] / "experiments" / "modal_smoke.py"
    spec = importlib.util.spec_from_file_location("modal_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_check_accepts_sdk_workspace_name_and_parenthesized_id(monkeypatch) -> None:
    modal_smoke = load_launcher()

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="Token: hidden\nWorkspace: expected-workspace (workspace-id)\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert modal_smoke._verify_profile("personal", "expected-workspace") is True


def test_remote_exception_handler_uses_cpu_message_sanitizer() -> None:
    import ast
    import inspect

    modal_smoke = load_launcher()
    tree = ast.parse(inspect.getsource(modal_smoke._remote_inference))
    handlers = [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)]
    assert any(
        any(
            isinstance(call, ast.Call)
            and ast.unparse(call.func) == "remote_smoke.sanitize_exception_message"
            for statement in handler.body
            for call in ast.walk(statement)
        )
        for handler in handlers
    )
