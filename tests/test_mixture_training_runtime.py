from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

from experiments import mixture_training_contracts as contracts
from experiments import mixture_training_runtime_execution as runtime_execution

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _first_party_module_path(module_name: str) -> str | None:
    if module_name == "experiments" or module_name.startswith("experiments."):
        stem = PROJECT_ROOT / Path(*module_name.split("."))
    elif module_name == "reflex_decisions" or module_name.startswith("reflex_decisions."):
        stem = PROJECT_ROOT / "src" / Path(*module_name.split("."))
    else:
        return None

    for candidate in (Path(str(stem) + ".py"), stem / "__init__.py"):
        if candidate.is_file():
            return str(candidate.relative_to(PROJECT_ROOT))
    return None


def _source_module_name(relative_path: str) -> str | None:
    if relative_path.startswith("experiments/"):
        module = relative_path.removesuffix(".py").replace("/", ".")
    elif relative_path.startswith("src/reflex_decisions/"):
        module = relative_path.removeprefix("src/").removesuffix(".py").replace("/", ".")
    else:
        return None
    return module.removesuffix(".__init__")


def _is_first_party_module(module_name: str) -> bool:
    return module_name in {"experiments", "reflex_decisions"} or module_name.startswith(
        ("experiments.", "reflex_decisions.")
    )


def test_runtime_training_constants_come_from_contracts() -> None:
    assert runtime_execution.LEARNING_RATE == contracts.LEARNING_RATE
    assert runtime_execution.WEIGHT_DECAY == contracts.WEIGHT_DECAY
    assert runtime_execution.MAX_GRADIENT_NORM == contracts.MAX_GRADIENT_NORM


def test_importing_runtime_on_cpu_does_not_import_model_or_modal_packages() -> None:
    script = """
import sys
from experiments import mixture_training_runtime
forbidden = ("torch", "transformers", "peft", "modal")
assert not any(
    name == prefix or name.startswith(prefix + ".")
    for name in sys.modules
    for prefix in forbidden
)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_source_fingerprint_allowlist_covers_local_import_closure() -> None:
    allowlisted_paths = set(contracts.SOURCE_FINGERPRINT_PATHS)
    missing: set[tuple[str, str, str]] = set()

    def require_module(importer: str, module_name: str) -> None:
        source_path = _first_party_module_path(module_name)
        if source_path is None:
            missing.add((importer, module_name, "<unresolved first-party module>"))
        elif source_path not in allowlisted_paths:
            missing.add((importer, module_name, source_path))

    for relative in sorted(allowlisted_paths):
        if not relative.endswith(".py"):
            continue
        source = PROJECT_ROOT / relative
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=relative)
        module = _source_module_name(relative)
        if module is None:
            continue
        package = module if relative.endswith("/__init__.py") else module.rpartition(".")[0]

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_first_party_module(alias.name):
                        require_module(relative, alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package_parts = package.split(".") if package else []
                    prefix = package_parts[: len(package_parts) - node.level + 1]
                    base = ".".join(prefix + (node.module.split(".") if node.module else []))
                else:
                    base = node.module or ""
                if _is_first_party_module(base):
                    require_module(relative, base)
                    base_path = _first_party_module_path(base)
                    if base_path is not None and base_path.endswith("/__init__.py"):
                        for alias in node.names:
                            if alias.name != "*":
                                candidate = f"{base}.{alias.name}"
                                if _first_party_module_path(candidate):
                                    require_module(relative, candidate)
            elif isinstance(node, ast.Call):
                is_dynamic_import = (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"import_module", "__import__"}
                ) or (isinstance(node.func, ast.Name) and node.func.id == "__import__")
                if (
                    is_dynamic_import
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and _is_first_party_module(node.args[0].value)
                ):
                    require_module(relative, node.args[0].value)

    assert not missing, (
        f"local imports are absent from the source fingerprint allowlist: {sorted(missing)}"
    )


def test_remote_source_fingerprints_import_an_isolated_uploaded_bundle(tmp_path) -> None:
    bundle_root = tmp_path / "uploaded-source"
    for relative in contracts.SOURCE_FINGERPRINT_PATHS:
        source = PROJECT_ROOT / relative
        destination = bundle_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    script = """
import sys
from pathlib import Path
from experiments import mixture_training_runtime

assert mixture_training_runtime._project_root() == Path.cwd()
measured = mixture_training_runtime._measure_remote_source_fingerprints()
assert set(measured) == set(mixture_training_runtime.core.SOURCE_FINGERPRINT_PATHS)
forbidden = ("torch", "torchvision", "transformers", "peft", "modal", "safetensors")
assert not any(
    name == prefix or name.startswith(prefix + ".")
    for name in sys.modules
    for prefix in forbidden
)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(bundle_root), str(bundle_root / "src")))
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=bundle_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
