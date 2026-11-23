"""Validation policy router: reparar / cuarentena / descartar.

The Pandera schema can raise on the first failing check, but in production we
want the *full* list of failures — that is what tells the operator which dirt
family hit the run. We use ``lazy=True`` so Pandera collects all failures and
then route each failing row into one of three buckets:

* ``valid``       — passes every check.
* ``quarantined`` — recoverable failures (e.g. unparseable ``signed_at``,
  phases that do not add up). Kept for owner review.
* ``discarded``   — fatal failures (e.g. negative ``total_amount``, malformed
  ``budget_id``, currency not allowed). Logged but not retried.

Schema-level failures (a column missing, an undeclared column, a wrong dtype)
carry no row index: they mean the *contract* broke, so every row is sent to
quarantine instead of being silently accepted.

The ``report`` dict is JSON-serializable so it can be written next to the
cleaned output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pandera.pandas as pa

from app.ingestion.cleaning.schemas import PHASES_SUM_CHECK_NAME, BudgetRecord

# Which failure types belong in quarantine vs. discard. Quarantine = the row
# CAN come back after manual review; discard = the row is structurally wrong.
QUARANTINE_CHECKS = {
    "not_nullable",                    # e.g. signed_at NaT after coercion
    PHASES_SUM_CHECK_NAME,             # phases do not add up to the total
}
DISCARD_CHECKS = {
    "in_range",                        # total_amount negative / too large
    "less_than_or_equal_to",
    "greater_than_or_equal_to",
    "isin",                            # currency not in {EUR,USD,GBP}
    "str_matches",                     # malformed budget_id or client_code
}


@dataclass
class ValidationResult:
    valid: pd.DataFrame
    quarantined: pd.DataFrame
    discarded: pd.DataFrame
    report: dict[str, Any] = field(default_factory=dict)
    # Row index → list of "column:check" failures. Lets the exporter attach
    # the reason to every quarantined/discarded record.
    reasons: dict[int, list[str]] = field(default_factory=dict)


def validate_with_policy(df: pd.DataFrame) -> ValidationResult:
    """Run :data:`BudgetRecord` against ``df`` and partition the rows."""
    if df.empty:
        return ValidationResult(
            valid=df.copy(),
            quarantined=df.iloc[0:0].copy(),
            discarded=df.iloc[0:0].copy(),
            report={"input_rows": 0},
        )

    try:
        validated = BudgetRecord.validate(df, lazy=True)
        return ValidationResult(
            valid=validated,
            quarantined=df.iloc[0:0].copy(),
            discarded=df.iloc[0:0].copy(),
            report={
                "input_rows": len(df),
                "valid_rows": len(validated),
                "quarantined_rows": 0,
                "discarded_rows": 0,
                "failures_by_check": {},
                "schema_level_failures": [],
            },
        )
    except pa.errors.SchemaErrors as err:
        return _route_failures(df, err)


def _route_failures(
    df: pd.DataFrame, err: pa.errors.SchemaErrors
) -> ValidationResult:
    failure_cases = err.failure_cases.copy()
    indices_quarantine: set[int] = set()
    indices_discard: set[int] = set()
    failures_by_check: dict[str, int] = {}
    schema_level: list[str] = []
    reasons: dict[int, list[str]] = {}
    seen: set[tuple[object, str]] = set()

    for _, row in failure_cases.iterrows():
        check_name = str(row.get("check", "unknown"))
        column = row.get("column")
        # Dataframe-level checks are reported once per column by Pandera;
        # they belong to the row, not to a column.
        is_frame_check = row.get("schema_context") == "DataFrameSchema" and (
            check_name == PHASES_SUM_CHECK_NAME
        )
        label = (
            f"{column}:{check_name}"
            if isinstance(column, str) and not is_frame_check
            else check_name
        )
        idx = row.get("index")
        key = (None if idx is None or pd.isna(idx) else idx, label)
        if key in seen:
            continue
        seen.add(key)
        failures_by_check[check_name] = failures_by_check.get(check_name, 0) + 1

        if idx is None or pd.isna(idx):
            schema_level.append(f"{label} ({row.get('failure_case')})")
            continue
        try:
            row_idx = int(idx)
        except (TypeError, ValueError):
            continue

        reasons.setdefault(row_idx, []).append(label)
        if any(disc in check_name for disc in DISCARD_CHECKS):
            indices_discard.add(row_idx)
        else:
            # Known quarantine checks AND unknown failure types — be safe,
            # quarantine for review rather than discard or accept.
            indices_quarantine.add(row_idx)

    if schema_level:
        # Contract broke: nothing can be trusted as "valid".
        all_rows = {int(i) for i in df.index}
        indices_quarantine |= all_rows - indices_discard
        for i in all_rows - indices_discard:
            reasons.setdefault(i, []).append("schema_level_failure")

    # Rows in discard set always lose, even if they also hit a quarantine check.
    indices_quarantine -= indices_discard
    discarded_mask = df.index.isin(indices_discard)
    quarantined_mask = df.index.isin(indices_quarantine)
    valid_mask = ~(discarded_mask | quarantined_mask)

    return ValidationResult(
        valid=df.loc[valid_mask].copy(),
        quarantined=df.loc[quarantined_mask].copy(),
        discarded=df.loc[discarded_mask].copy(),
        report={
            "input_rows": len(df),
            "valid_rows": int(valid_mask.sum()),
            "quarantined_rows": int(quarantined_mask.sum()),
            "discarded_rows": int(discarded_mask.sum()),
            "failures_by_check": failures_by_check,
            "schema_level_failures": schema_level,
        },
        reasons=reasons,
    )
