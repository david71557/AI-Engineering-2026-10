"""Cleaning and validation layer.

Two passes:

* ``budget_records.clean_budget_records`` — pandas-level fixes (null
  placeholders, currency casing, permissive coercion, hash-based dedup).
* ``policy.validate_with_policy`` — Pandera-level field/cross-column checks,
  routing failures into valid / quarantined / discarded partitions.

The split is deliberate: cleaning shapes the data, validation gatekeeps it.
"""
from app.ingestion.cleaning.budget_records import clean_budget_records
from app.ingestion.cleaning.policy import ValidationResult, validate_with_policy
from app.ingestion.cleaning.schemas import BudgetRecord

# ``export`` is NOT re-exported here: it imports the catalog/loader layers and
# would create an import cycle for callers that only need cleaning.

__all__ = [
    "BudgetRecord",
    "ValidationResult",
    "clean_budget_records",
    "validate_with_policy",
]
