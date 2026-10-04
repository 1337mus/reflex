"""Immutable request schema and semantic request digests."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _require_nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class Option(BaseModel):
    """One semantic answer available to a decision request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    label: str
    description: str | None = None

    @field_validator("id", "label")
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        return _require_nonblank(value)


class DecisionRequest(BaseModel):
    """A closed set of semantic options for one decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    context: str
    question: str
    options: tuple[Option, ...] = Field(min_length=2, max_length=16)

    @field_validator("context", "question")
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        return _require_nonblank(value)

    @model_validator(mode="after")
    def require_unique_ids(self) -> DecisionRequest:
        ids = [option.id for option in self.options]
        if len(ids) != len(set(ids)):
            raise ValueError("option IDs must be unique")
        return self

    @property
    def schema_hash(self) -> str:
        return _digest(self._schema_payload())

    @property
    def request_hash(self) -> str:
        return _digest({"context": self.context, **self._schema_payload()})

    def _schema_payload(self) -> dict[str, Any]:
        options = [option.model_dump(mode="json") for option in self.options]
        return {
            "question": self.question,
            "options": sorted(options, key=lambda option: option["id"]),
        }
