"""Pandera v2 schema for the cleaned budget DataFrame.

Note the Pandera v2 import path: ``import pandera.pandas as pa``. In v1 it was
``import pandera as pa``. The migration broke many tutorials.

The schema mixes per-column constraints with one cross-column dataframe check
that enforces ``sum(phases.amount) == total_amount``. Every column the cleaning
step produces is declared (``strict=True``): an undeclared column is a real
contract break, not noise to ignore.
"""
from __future__ import annotations

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

BUDGET_ID_PATTERN = r"^BUDGET-\d{4}-\d{4}$"
CLIENT_CODE_PATTERN = r"^CLI-\d{4}$"
ALLOWED_CURRENCIES = ["EUR", "USD", "GBP"]
# Absolute tolerance (in currency units) when comparing phases vs. total.
PHASES_SUM_TOLERANCE = 0.01

PHASES_SUM_CHECK_NAME = "phases_sum_matches_total"


def _phases_sum_matches_total(df: pd.DataFrame) -> pd.Series:
    """Row-wise: True when the phases add up to ``total_amount``.

    Budgets without phases cannot be checked and pass (absence of detail is
    not an inconsistency). A phase with a missing amount fails the row.
    """
    if "phases" not in df.columns or "total_amount" not in df.columns:
        return pd.Series(True, index=df.index)

    def _row_ok(row: pd.Series) -> bool:
        phases = row["phases"]
        if not isinstance(phases, list) or not phases:
            return True
        amounts = [p.get("amount") for p in phases]
        if any(a is None for a in amounts) or pd.isna(row["total_amount"]):
            return False
        return abs(sum(amounts) - row["total_amount"]) <= PHASES_SUM_TOLERANCE

    return df.apply(_row_ok, axis=1).astype(bool)


BudgetRecord: DataFrameSchema = DataFrameSchema(
    columns={
        "budget_id": Column(
            str,
            checks=Check.str_matches(BUDGET_ID_PATTERN),
            nullable=False,
            required=True,
        ),
        "client_name": Column(
            str,
            nullable=True,  # cleaning maps TBD/N/A to NA — kept, flagged downstream
            required=True,
        ),
        "client_code": Column(
            str,
            checks=Check.str_matches(CLIENT_CODE_PATTERN),
            nullable=False,
            required=True,
        ),
        "currency": Column(
            str,
            checks=Check.isin(ALLOWED_CURRENCIES),
            nullable=False,
            required=True,
        ),
        "total_amount": Column(
            float,
            checks=[Check.ge(0), Check.le(10_000_000)],
            nullable=False,
            required=True,
        ),
        "signed_at": Column(
            "datetime64[ns]",
            nullable=False,  # unparseable date → NaT → quarantine for review
            required=True,
        ),
        # Optional columns: declared so ``strict=True`` only flags true strays.
        "phases": Column(object, nullable=True, required=False),
        "contact": Column(str, nullable=True, required=False),
        "contact_email": Column(str, nullable=True, required=False),
        "notes": Column(str, nullable=True, required=False),
        "content_hash": Column(str, nullable=False, required=False),
        "source_file": Column(str, nullable=True, required=False),
    },
    checks=[
        # No ``error=``: Pandera would report that text instead of ``name``
        # in ``failure_cases.check``, and the policy routes on the name.
        Check(_phases_sum_matches_total, name=PHASES_SUM_CHECK_NAME)
    ],
    strict=True,
    coerce=False,
)
