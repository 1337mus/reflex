"""Fixed contract for the synthetic tool-choice dataset."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .data import DecisionRecord, SplitName
from .tool_rules import ToolScenario

VERSION = "tool-data-v1"
SOURCE_ID = "synthetic-tool-choice-v1"
FAMILY = "tool-choice"
SPLITS: tuple[SplitName, ...] = ("train", "development", "calibration", "test")
MAX_COUNTS = (56, 14, 14, 28)
TEMPLATE_IDS: Mapping[SplitName, str] = MappingProxyType(
    {
        "train": "tool-v1",
        "development": "tool-development-v1",
        "calibration": "tool-calibration-v1",
        "test": "tool-sealed-v1",
    }
)
CASE_KINDS = (
    "complete_tool",
    "complete_no_eligible",
    "missing_tool_agree",
    "missing_none_agree",
    "missing_tool_conflict",
    "missing_tool_vs_none",
    "multiple_missing",
)


@dataclass(frozen=True)
class ToolVocabulary:
    capabilities: tuple[str, ...]
    input_types: tuple[str, ...]
    permissions: tuple[str, ...]
    tool_ids: tuple[str, ...]


@dataclass(frozen=True)
class ToolDatum:
    split: SplitName
    scenario: ToolScenario
    record: DecisionRecord
    template_id: str
    case_kind: str


VOCABULARIES: Mapping[SplitName, ToolVocabulary] = MappingProxyType(
    {
        "train": ToolVocabulary(
            capabilities=("lookup", "summarize", "translate", "extract"),
            input_types=("memo", "photo", "clip"),
            permissions=("private", "external"),
            tool_ids=("scout", "scribe", "bridge", "lens", "tally", "vault"),
        ),
        "development": ToolVocabulary(
            capabilities=("classify", "compare", "transcribe", "verify"),
            input_types=("invoice", "diagram", "recording"),
            permissions=("restricted", "remote"),
            tool_ids=("beacon", "ledger", "parley", "keen", "matrix", "quartz"),
        ),
        "calibration": ToolVocabulary(
            capabilities=("merge", "archive", "caption", "review"),
            input_types=("notice", "sketch", "stream"),
            permissions=("sealed", "network"),
            tool_ids=("fable", "grove", "harbor", "isotope", "juniper", "kepler"),
        ),
        "test": ToolVocabulary(
            capabilities=("route", "compress", "localize", "index"),
            input_types=("prompt", "portrait", "audio"),
            permissions=("confidential", "crossborder"),
            tool_ids=("lynx", "mosaic", "nimbus", "orbit", "pioneer", "quill"),
        ),
    }
)
