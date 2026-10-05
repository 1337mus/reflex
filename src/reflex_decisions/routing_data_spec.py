"""Fixed contract for the synthetic support-routing corpus."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .data import DecisionRecord, SplitName
from .routing_rules import Feature, RoutingScenario

VERSION = "routing-data-v1"
SOURCE_ID = "synthetic-support-routing-v1"
FAMILY = "support-routing"
SPLITS: tuple[SplitName, ...] = ("train", "development", "calibration", "test")
TEMPLATE_IDS: Mapping[SplitName, str] = MappingProxyType(
    {
        "train": "routing-v1",
        "development": "routing-development-v1",
        "calibration": "routing-calibration-v1",
        "test": "routing-sealed-v1",
    }
)
CASE_KINDS = (
    "complete_route",
    "complete_no_route",
    "missing_route_agree",
    "missing_no_route_agree",
    "missing_route_conflict",
    "missing_route_vs_none",
    "multiple_missing",
)


@dataclass(frozen=True)
class RoutingVocabulary:
    features: tuple[Feature, ...]
    destination_ids: tuple[str, ...]


@dataclass(frozen=True)
class RoutingDatum:
    split: SplitName
    scenario: RoutingScenario
    record: DecisionRecord
    template_id: str
    case_kind: str


VOCABULARIES: Mapping[SplitName, RoutingVocabulary] = MappingProxyType(
    {
        "train": RoutingVocabulary(
            (
                Feature("priority", ("routine", "urgent")),
                Feature("region", ("east", "west")),
                Feature("channel", ("email", "phone")),
                Feature("account", ("individual", "business")),
            ),
            ("cedar", "birch", "maple", "oak", "pine", "elm"),
        ),
        "development": RoutingVocabulary(
            (
                Feature("plan", ("starter", "premium")),
                Feature("device", ("desktop", "mobile")),
                Feature("language", ("english", "spanish")),
                Feature("connection", ("wired", "wireless")),
            ),
            ("amber", "cobalt", "jade", "ruby", "pearl", "onyx"),
        ),
        "calibration": RoutingVocabulary(
            (
                Feature("status", ("new", "renewal")),
                Feature("category", ("billing", "technical")),
                Feature("period", ("day", "night")),
                Feature("age", ("recent", "established")),
            ),
            ("falcon", "heron", "ibis", "kestrel", "lark", "osprey"),
        ),
        "test": RoutingVocabulary(
            (
                Feature("delivery", ("parcel", "freight")),
                Feature("material", ("paper", "metal")),
                Feature("service", ("onsite", "remote")),
                Feature("container", ("crate", "envelope")),
            ),
            ("atlas", "boreal", "comet", "delta", "equinox", "finch"),
        ),
    }
)
