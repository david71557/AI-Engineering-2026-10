"""Export the cleaned historical budgets as structured JSON.

This is the closing step of Session 06: the output of
``clean_budget_records`` + ``validate_with_policy`` is turned into four files
under ``PROCESSED_DATA_DIR``:

* ``budgets_clean.json``      — valid budgets. Every record is validated
  against :class:`CleanBudget` (Pydantic) before it is written, so the file
  has a fixed, documented shape that downstream steps (embedding, retrieval,
  few-shot examples) can rely on.
* ``budgets_quarantine.json`` — recoverable rows, with the failing checks.
* ``budgets_discarded.json``  — fatal rows, with the failing checks.
* ``cleaning_report.json``    — counts, failures by check, lineage.

Records enter through the catalog: only an ``include`` source can be exported,
the same rule the ingestion endpoint enforces. Free-text PII fields can be
pseudonymized on the way out (``ConsistentPseudonymizer``).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.ingestion.catalog.models import CatalogDecision, DataCatalog
from app.ingestion.cleaning.budget_records import clean_budget_records
from app.ingestion.cleaning.policy import validate_with_policy
from app.ingestion.cleaning.schemas import (
    ALLOWED_CURRENCIES,
    BUDGET_ID_PATTERN,
    CLIENT_CODE_PATTERN,
)
from app.ingestion.loaders.filesystem import FileSystemLoader

EXPORT_SCHEMA_VERSION = "1.0.0"
# Structured fields whose WHOLE value is PII → replaced as a unit, no NER.
PII_WHOLE_VALUE_FIELDS = {"contact": "PERSON", "contact_email": "EMAIL_ADDRESS"}
# Free-text fields scanned with Presidio, restricted to pattern-based entities.
# PERSON is deliberately excluded: spaCy NER on short business notes yields
# false positives ("Carga inversa…" → "Carga" tagged as a person) that corrupt
# the text. Names already known from the record are replaced literally instead.
PII_FREE_TEXT_FIELDS = ("notes",)
PII_FREE_TEXT_ENTITIES = ["EMAIL_ADDRESS"]

OUTPUT_FILES = {
    "clean": "budgets_clean.json",
    "quarantine": "budgets_quarantine.json",
    "discarded": "budgets_discarded.json",
    "report": "cleaning_report.json",
}


class ExportRejected(Exception):
    """The source is unknown or not ``include`` in the catalog."""


# --- Output contract ----------------------------------------------------------


class CleanPhase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    weeks: float | None = Field(default=None, ge=0)
    amount: float = Field(ge=0)


class CleanBudget(BaseModel):
    """One cleaned, normalized, validated historical budget."""

    model_config = ConfigDict(extra="forbid")

    budget_id: str = Field(pattern=BUDGET_ID_PATTERN)
    client_name: str | None
    client_code: str = Field(pattern=CLIENT_CODE_PATTERN)
    currency: Literal[tuple(ALLOWED_CURRENCIES)]  # type: ignore[valid-type]
    total_amount: float = Field(ge=0)
    signed_at: date
    phases: list[CleanPhase] = Field(default_factory=list)
    total_weeks: float | None = None
    contact: str | None = None
    contact_email: str | None = None
    notes: str | None = None
    content_hash: str
    source_file: str | None = None


class RejectedBudget(BaseModel):
    """A row that did not reach ``budgets_clean.json``, with the reason."""

    budget_id: str | None
    bucket: Literal["quarantine", "discarded"]
    reasons: list[str]
    record: dict[str, Any]


@dataclass
class ExportBundle:
    clean: list[CleanBudget]
    quarantined: list[RejectedBudget]
    discarded: list[RejectedBudget]
    report: dict[str, Any] = field(default_factory=dict)


# --- Pipeline -----------------------------------------------------------------


def load_budget_records(
    catalog: DataCatalog, source_name: str, loader: FileSystemLoader
) -> list[dict]:
    """Read every raw budget of an ``include`` source, tagging ``source_file``."""
    source = catalog.find(source_name)
    if source is None:
        raise ExportRejected(f"source {source_name!r} not found in catalog")
    if source.decision is not CatalogDecision.INCLUDE:
        raise ExportRejected(
            f"source {source_name!r} has decision={source.decision.value!r}; "
            "only 'include' sources can be exported"
        )
    records: list[dict] = []
    for blob in loader.iter_blobs(source.location, {source.format}):
        record = json.loads(blob.bytes_.decode("utf-8"))
        record["source_file"] = Path(blob.relative_path).as_posix()
        records.append(record)
    return records


def build_export(
    records: list[dict],
    *,
    source_name: str,
    source_version: str,
    pseudonymizer: Any | None = None,
) -> ExportBundle:
    """Clean → validate → (pseudonymize) → typed export bundle."""
    df = clean_budget_records(records)
    result = validate_with_policy(df)

    clean: list[CleanBudget] = []
    contract_failures: list[RejectedBudget] = []
    for _, row in result.valid.iterrows():
        record = _finalize(_row_to_dict(row), pseudonymizer)
        try:
            clean.append(CleanBudget.model_validate(record))
        except ValidationError as exc:
            # Passed Pandera but not the output contract (e.g. a phase with
            # no name or a negative phase amount) → review, never silently.
            contract_failures.append(
                RejectedBudget(
                    budget_id=record.get("budget_id"),
                    bucket="quarantine",
                    reasons=[
                        "contract:" + ".".join(str(p) for p in e["loc"]) + ":" + e["type"]
                        for e in exc.errors()
                    ],
                    record=record,
                )
            )
    quarantined = contract_failures + _rejected(
        result.quarantined, "quarantine", result.reasons, pseudonymizer
    )
    discarded = _rejected(result.discarded, "discarded", result.reasons, pseudonymizer)

    report = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_name": source_name,
        "source_version": source_version,
        "pseudonymized": pseudonymizer is not None,
        "files_read": len(records),
        "rows_after_dedup": len(df),
        "duplicates_removed": len(records) - len(df),
        **result.report,
        # Final counts after the output contract (may move valid → quarantine).
        "exported_clean": len(clean),
        "exported_quarantine": len(quarantined),
        "exported_discarded": len(discarded),
        "rejections": {
            r.budget_id or "?": {"bucket": r.bucket, "reasons": r.reasons}
            for r in quarantined + discarded
        },
    }
    return ExportBundle(
        clean=clean, quarantined=quarantined, discarded=discarded, report=report
    )


def write_export(bundle: ExportBundle, out_dir: Path | str) -> dict[str, Path]:
    """Write the four JSON files under ``out_dir``; return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "generated_at": bundle.report.get("generated_at"),
        "source_name": bundle.report.get("source_name"),
        "source_version": bundle.report.get("source_version"),
        "pseudonymized": bundle.report.get("pseudonymized", False),
    }
    payloads = {
        "clean": {
            **meta,
            "count": len(bundle.clean),
            "budgets": [b.model_dump(mode="json") for b in bundle.clean],
        },
        "quarantine": {
            **meta,
            "count": len(bundle.quarantined),
            "budgets": [r.model_dump(mode="json") for r in bundle.quarantined],
        },
        "discarded": {
            **meta,
            "count": len(bundle.discarded),
            "budgets": [r.model_dump(mode="json") for r in bundle.discarded],
        },
        "report": bundle.report,
    }
    paths: dict[str, Path] = {}
    for key, payload in payloads.items():
        path = out / OUTPUT_FILES[key]
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        paths[key] = path
    return paths


# --- Helpers ------------------------------------------------------------------


def _rejected(
    df: pd.DataFrame,
    bucket: Literal["quarantine", "discarded"],
    reasons: dict[int, list[str]],
    pseudonymizer: Any | None,
) -> list[RejectedBudget]:
    out: list[RejectedBudget] = []
    for idx, row in df.iterrows():
        record = _finalize(_row_to_dict(row), pseudonymizer)
        out.append(
            RejectedBudget(
                budget_id=record.get("budget_id"),
                bucket=bucket,
                reasons=reasons.get(int(idx), []),
                record=record,
            )
        )
    return out


def _finalize(record: dict[str, Any], pseudonymizer: Any | None) -> dict[str, Any]:
    """Derived fields + optional pseudonymization of free-text PII."""
    phases = record.get("phases") or []
    weeks = [p.get("weeks") for p in phases]
    record["total_weeks"] = (
        sum(weeks) if phases and all(w is not None for w in weeks) else None
    )
    if pseudonymizer is not None:
        _pseudonymize_record(record, pseudonymizer)
    return record


def _pseudonymize_record(record: dict[str, Any], pseudonymizer: Any) -> None:
    """Whole-value replacement for structured PII, pattern-only for free text.

    Known originals (the record's own contact name/email) are also replaced
    literally inside the free text, so "Laura Fernández confirmó…" in a note
    gets the same pseudonym as the ``contact`` field, without relying on NER.
    """
    known: dict[str, str] = {}
    for key, entity_type in PII_WHOLE_VALUE_FIELDS.items():
        value = record.get(key)
        if isinstance(value, str) and value:
            pseudonym = pseudonymizer.pseudonymize_value(value, entity_type)
            known[value] = pseudonym
            record[key] = pseudonym

    for key in PII_FREE_TEXT_FIELDS:
        text = record.get(key)
        if not isinstance(text, str) or not text:
            continue
        # Order matters: scan the ORIGINAL text first. Replacing known values
        # first would put pseudonym emails in the text, and the email scan
        # would then pseudonymize the pseudonyms. Hashing the original value
        # yields the same pseudonym as the structured field (shared store).
        text = pseudonymizer.pseudonymize(
            text, entities=PII_FREE_TEXT_ENTITIES
        ).pseudonymized_text
        # Then known originals the scan cannot find (names). Longest first so
        # "Laura Fernández" wins over a shorter overlapping value.
        for original in sorted(known, key=len, reverse=True):
            text = text.replace(original, known[original])
        record[key] = text


def _row_to_dict(row: pd.Series) -> dict[str, Any]:
    return {key: _json_value(value) for key, value in row.items()}


def _json_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_json_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "item"):  # numpy scalar → python
        return value.item()
    return value
