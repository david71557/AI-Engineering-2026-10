"""Tests for the Session 06 closing step: cleaned budgets exported as JSON."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from app.ingestion.catalog.loader import load_catalog
from app.ingestion.cleaning.budget_records import clean_budget_records
from app.ingestion.cleaning.export import (
    OUTPUT_FILES,
    CleanBudget,
    ExportRejected,
    build_export,
    load_budget_records,
    write_export,
)
from app.ingestion.cleaning.policy import validate_with_policy
from app.ingestion.loaders.filesystem import FileSystemLoader

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "catalog" / "catalog.yaml"
SEED = ROOT / "data" / "seed"


def _record(**overrides):
    base = {
        "budget_id": "BUDGET-2024-0001",
        "client_name": "Acme",
        "client_code": "CLI-0001",
        "currency": "EUR",
        "total_amount": 1000,
        "signed_at": "2024-01-01",
        "phases": [{"name": "Build", "weeks": 2, "amount": 1000}],
    }
    base.update(overrides)
    return base


@pytest.fixture
def seed_records():
    return load_budget_records(
        load_catalog(CATALOG), "presupuestos_json", FileSystemLoader(SEED)
    )


# --- Regressions fixed vs. the estimator reference -------------------------


def test_mixed_date_formats_are_all_parsed():
    """pandas 2 infers the format from row 0; per-value parsing must not NaT
    the Spanish date when an ISO date comes first."""
    df = clean_budget_records(
        [
            _record(budget_id="BUDGET-2024-0001", signed_at="2024-02-14"),
            _record(budget_id="BUDGET-2024-0002", signed_at="12/03/2024"),
        ]
    )
    dates = dict(zip(df["budget_id"], df["signed_at"]))
    assert dates["BUDGET-2024-0002"].month == 3
    assert dates["BUDGET-2024-0002"].day == 12
    assert validate_with_policy(df).report["valid_rows"] == 2


def test_integer_amounts_do_not_trigger_dtype_failure():
    result = validate_with_policy(clean_budget_records([_record(total_amount=1000)]))
    assert result.report["failures_by_check"] == {}


def test_phases_not_adding_up_are_quarantined():
    df = clean_budget_records(
        [_record(total_amount=1000, phases=[{"name": "A", "weeks": 1, "amount": 400}])]
    )
    result = validate_with_policy(df)
    assert len(result.quarantined) == 1
    assert "phases_sum_matches_total" in result.reasons[0]


def test_unparseable_date_is_quarantined_not_discarded():
    result = validate_with_policy(clean_budget_records([_record(signed_at="ayer")]))
    assert len(result.quarantined) == 1
    assert len(result.discarded) == 0


def test_undeclared_column_quarantines_whole_batch():
    df = clean_budget_records([_record(), _record(budget_id="BUDGET-2024-0002")])
    df["unexpected"] = "x"
    result = validate_with_policy(df)
    assert result.report["schema_level_failures"]
    assert len(result.valid) == 0
    assert len(result.quarantined) == 2


# --- Export pipeline ---------------------------------------------------------


def test_seed_corpus_story(seed_records):
    """6 files in, BUDGET-2024-0005 collapses to v2, -50000 is discarded."""
    bundle = build_export(
        seed_records, source_name="presupuestos_json", source_version="1.0.0"
    )
    assert bundle.report["files_read"] == 6
    assert bundle.report["duplicates_removed"] == 1
    assert [b.budget_id for b in bundle.clean] == [
        "BUDGET-2024-0001",
        "BUDGET-2024-0002",
        "BUDGET-2024-0003",
        "BUDGET-2024-0005",
    ]
    assert [r.budget_id for r in bundle.discarded] == ["BUDGET-2024-0004"]
    assert bundle.quarantined == []

    by_id = {b.budget_id: b for b in bundle.clean}
    assert by_id["BUDGET-2024-0002"].currency == "EUR"          # eur → EUR
    assert by_id["BUDGET-2024-0002"].signed_at.isoformat() == "2024-03-12"
    assert by_id["BUDGET-2024-0003"].client_name is None        # TBD → null
    assert by_id["BUDGET-2024-0003"].contact_email is None      # "" → null
    assert by_id["BUDGET-2024-0005"].total_amount == 32000      # latest version
    assert by_id["BUDGET-2024-0005"].source_file.endswith("0005-v2.json")


def test_written_files_are_structured_json(seed_records, tmp_path):
    bundle = build_export(
        seed_records, source_name="presupuestos_json", source_version="1.0.0"
    )
    paths = write_export(bundle, tmp_path)
    assert set(paths) == set(OUTPUT_FILES)

    clean = json.loads(paths["clean"].read_text(encoding="utf-8"))
    assert clean["schema_version"] == "1.0.0"
    assert clean["count"] == len(clean["budgets"]) == 4
    for item in clean["budgets"]:
        CleanBudget.model_validate(item)  # round-trips through the contract

    discarded = json.loads(paths["discarded"].read_text(encoding="utf-8"))
    assert discarded["budgets"][0]["reasons"] == [
        "total_amount:greater_than_or_equal_to(0)"
    ]
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["exported_clean"] == 4
    # Accents survive (ensure_ascii=False + utf-8).
    assert "Acme España" in paths["clean"].read_text(encoding="utf-8")


def test_export_only_accepts_included_sources():
    catalog = load_catalog(CATALOG)
    with pytest.raises(ExportRejected):
        load_budget_records(catalog, "rate_card_xlsx", FileSystemLoader(SEED))
    with pytest.raises(ExportRejected):
        load_budget_records(catalog, "does_not_exist", FileSystemLoader(SEED))


@dataclass
class _Span:
    entity_type: str
    start: int
    end: int
    score: float = 0.85


class _NaiveNerAnalyzer:
    """Mimics spaCy's real false positive: any capitalised 'Carga' is a PERSON.
    Also finds emails, like Presidio's pattern recognizer."""

    def __init__(self):
        self.calls: list[tuple[str, list[str] | None]] = []

    def analyze(self, *, text, language, entities=None):
        import re

        self.calls.append((text, entities))
        found = [
            _Span("EMAIL_ADDRESS", m.start(), m.end())
            for m in re.finditer(r"[\w.]+@[\w.-]+\.\w+", text)
        ]
        found += [_Span("PERSON", m.start(), m.end()) for m in re.finditer(r"Carga", text)]
        wanted = set(entities) if entities else None
        return [s for s in found if wanted is None or s.entity_type in wanted]


def _real_pseudonymizer(analyzer):
    from app.ingestion.pii import ConsistentPseudonymizer, InMemoryMappingStore

    return ConsistentPseudonymizer(
        analyzer=analyzer,
        mapping_store=InMemoryMappingStore(),
        salt="unit-test-salt",
    )


def _export(records, pseudonymizer):
    return build_export(
        records,
        source_name="presupuestos_json",
        source_version="1.0.0",
        pseudonymizer=pseudonymizer,
    )


def test_structured_pii_replaced_as_whole_value(seed_records):
    bundle = _export(seed_records, _real_pseudonymizer(_NaiveNerAnalyzer()))
    assert bundle.report["pseudonymized"] is True
    first = bundle.clean[0]
    assert first.contact and first.contact != "Laura Fernández"
    assert first.contact_email and "acme.es" not in first.contact_email
    # Discarded rows are written to disk too → also pseudonymized.
    assert bundle.discarded[0].record["contact"] != "Pilar Alonso"


def test_notes_are_not_corrupted_by_person_ner(seed_records):
    """Regression: 'Carga inversa…' must survive — PERSON NER is not run on notes."""
    analyzer = _NaiveNerAnalyzer()
    bundle = _export(seed_records, _real_pseudonymizer(analyzer))
    note = bundle.discarded[0].record["notes"]
    assert note.startswith("Carga inversa por nota de abono")
    assert all(entities == ["EMAIL_ADDRESS"] for _, entities in analyzer.calls)


def test_known_contact_inside_notes_gets_same_pseudonym():
    records = [
        _record(
            contact="Laura Fernández",
            contact_email="laura@acme.es",
            notes="Laura Fernández pide copia a laura@acme.es. Carga urgente.",
        )
    ]
    bundle = _export(records, _real_pseudonymizer(_NaiveNerAnalyzer()))
    budget = bundle.clean[0]
    assert "Laura Fernández" not in budget.notes
    assert "laura@acme.es" not in budget.notes
    assert budget.notes.startswith(budget.contact)          # same pseudonym
    assert budget.contact_email in budget.notes             # same pseudonym
    assert budget.notes.endswith("Carga urgente.")          # business text intact


def test_pseudonyms_consistent_across_runs_with_shared_store(seed_records):
    from app.ingestion.pii import ConsistentPseudonymizer, InMemoryMappingStore

    store = InMemoryMappingStore()

    def make():
        return ConsistentPseudonymizer(
            analyzer=_NaiveNerAnalyzer(), mapping_store=store, salt="s"
        )

    a = _export(seed_records, make())
    b = _export(seed_records, make())
    assert [x.contact for x in a.clean] == [x.contact for x in b.clean]
