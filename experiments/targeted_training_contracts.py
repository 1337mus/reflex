"""Frozen constants for the targeted matched-training runner."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.targeted_training_core import ROLE_CONTROL, ROLE_TREATMENT, ROLE_UNCHANGED

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
SELECTED_ADAPTER_TENSOR_SHA256 = "b9ade96b9f6077934985a4b261a6f7400a1e210003b02094b492f36c8a844324"
TOKENIZER_FILE_SHA256 = {
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    "tokenizer.json": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
    "tokenizer_config.json": "e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de",
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
}
TRAINING_SEED = 20261014
LEARNING_RATE = 5e-5
BETAS = (0.9, 0.999)
EPSILON = 1e-8
WEIGHT_DECAY = 0.0
MAX_GRADIENT_NORM = 1.0
MICROBATCHES_PER_UPDATE = 4
UPDATES = 1200
PROFILE = "reflex-personal"
WORKSPACE = "rajath-61258"
ENVIRONMENT = "main"
MODAL_SDK_VERSION = "1.6.1"
BUNDLE_MANIFEST_SHA256 = "104a2a379d761307e189095d93be36cd60830e061f2012804bdf354b6349abe1"


@dataclass(frozen=True)
class RoleResources:
    timeout_seconds: int
    startup_timeout_seconds: int = 300
    cpu: int = 2
    memory_mib: int = 16 * 1024
    max_containers: int = 1
    max_inputs: int = 1
    scaledown_window_seconds: int = 2
    retries: int = 0


ROLE_RESOURCES = {
    ROLE_UNCHANGED: RoleResources(timeout_seconds=1800),
    ROLE_CONTROL: RoleResources(timeout_seconds=5400),
    ROLE_TREATMENT: RoleResources(timeout_seconds=5400),
}
