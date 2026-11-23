"""Reparable cleaning for the budget tabular view.

Steps, applied to a DataFrame whose schema we treat as **raw**:

1. **Null placeholders** — ``"TBD"``, ``"N/A"``, empty strings → ``pd.NA``.
   Disguised nulls are the easiest dirt to miss and the hardest to debug
   downstream; we normalize at the boundary. Surviving strings are stripped.

2. **Currency casing** — ``EUR`` vs ``eur`` collapses to upper. Any other
   currency is preserved (validation will catch it).

3. **Date coercion** — ``signed_at`` can come as ISO ``2024-05-02`` or as the
   Spanish ``12/03/2024``. Each value is parsed on its own against an explicit
   list of formats. A single vectorized ``pd.to_datetime`` call is NOT safe
   here: pandas 2 infers the format from the first row and silently turns
   every row in a different format into ``NaT``.

4. **Numeric coercion** — ``total_amount`` and ``phases[].amount`` coerced to
   ``float`` (strings sneak in from manual exports); unparseable → ``NaN``.
   Emails are lower-cased.

5. **Hash-based dedup** — duplicates are detected by content hash. When the
   same ``budget_id`` appears twice with different content, the *policy*
   (declared here in code) is "keep the row with the most recent
   ``signed_at``". That is a business decision, not a technical one —
   documenting it is part of the deliverable.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Iterable

import pandas as pd

NULL_PLACEHOLDERS = {"TBD", "N/A", "n/a", "tbd", "", "null", "None", "-"}
TEXT_COLUMNS = ("client_name", "contact", "contact_email", "notes")
# Accepted date formats, tried in order. ISO first; then Spanish day-first.
DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d")


def clean_budget_records(records: Iterable[dict]) -> pd.DataFrame:
    """Return a cleaned DataFrame from the raw list of budget dicts.

    The input is the parsed JSON content — *not* the ``Document.text`` rendered
    markdown. The cleaning step works on the original record, before any
    embedding takes place.
    """
    df = pd.DataFrame(list(records))
    if df.empty:
        return df

    # 1. Null placeholders (+ whitespace strip)
    for column in TEXT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].apply(_normalize_text)

    # 2. Currency casing
    if "currency" in df.columns:
        df["currency"] = df["currency"].astype("string").str.strip().str.upper()

    # 3. Date coercion (per value, explicit formats)
    if "signed_at" in df.columns:
        df["signed_at"] = pd.to_datetime(
            df["signed_at"].apply(parse_date), errors="coerce"
        ).astype("datetime64[ns]")

    # 4. Numeric coercion
    if "total_amount" in df.columns:
        df["total_amount"] = pd.to_numeric(
            df["total_amount"], errors="coerce"
        ).astype("float64")
    if "phases" in df.columns:
        df["phases"] = df["phases"].apply(_normalize_phases)
    if "contact_email" in df.columns:
        df["contact_email"] = df["contact_email"].apply(
            lambda v: v.lower() if isinstance(v, str) else v
        )

    # 5. Hash-based dedup with explicit business rule: keep latest signed_at.
    df["content_hash"] = df.apply(_content_hash, axis=1)
    if {"budget_id", "signed_at"}.issubset(df.columns):
        # Identical content is always redundant, whatever its id.
        df = df.drop_duplicates(subset=["content_hash"], keep="last")
        # Sort so the latest signed_at lands LAST per budget_id, then drop
        # earlier rows. Reset index so downstream callers see no gaps.
        df = df.sort_values(
            by=["budget_id", "signed_at"], na_position="first", kind="stable"
        )
        df = df.drop_duplicates(subset=["budget_id"], keep="last")
    return df.reset_index(drop=True)


def parse_date(value: Any) -> datetime | None:
    """Parse a single date value against :data:`DATE_FORMATS`; ``None`` if none fit."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _normalize_text(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        return pd.NA if stripped in NULL_PLACEHOLDERS else stripped
    return value


def _normalize_phases(value: Any) -> list[dict]:
    """Return a list of ``{name, weeks, amount}`` with numeric coercion."""
    if not isinstance(value, list):
        return []
    phases: list[dict] = []
    for phase in value:
        if not isinstance(phase, dict):
            continue
        phases.append(
            {
                "name": str(phase.get("name", "")).strip() or None,
                "weeks": _to_number(phase.get("weeks")),
                "amount": _to_number(phase.get("amount")),
            }
        )
    return phases


def _to_number(value: Any) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def _content_hash(row: pd.Series) -> str:
    """Canonical JSON of the row → sha256.

    Missing values are normalized to ``None`` so two semantically identical
    rows that differ only in dtype produce the same hash. ``source_file`` is
    lineage, not content: it is excluded so copies of a file dedupe.
    """
    payload = {
        k: _serializable(v)
        for k, v in row.items()
        if k not in {"content_hash", "source_file"}
    }
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _serializable(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value
