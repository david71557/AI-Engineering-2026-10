#!/usr/bin/env python3
"""Session 06 closing step — export cleaned historical budgets as JSON.

Pipeline: catalog (only ``include`` sources) → FileSystemLoader → cleaning →
Pandera validation policy → (optional) pseudonymization → Pydantic contract →
``data/processed/*.json``.

Usage::

    uv run python scripts/export_clean_budgets.py
    uv run python scripts/export_clean_budgets.py --pseudonymize memory
    uv run python scripts/export_clean_budgets.py --pseudonymize postgres  # needs DB
    # or, inside docker:
    docker compose exec estimator python scripts/export_clean_budgets.py

``--pseudonymize memory`` keeps pseudonyms consistent within one run only;
``postgres`` persists the mapping table so pseudonyms are stable across runs
(and reversible / forgettable per GDPR Art. 17).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.ingestion.catalog import load_catalog  # noqa: E402
from app.ingestion.cleaning.export import (  # noqa: E402
    build_export,
    load_budget_records,
    write_export,
)
from app.ingestion.loaders.filesystem import FileSystemLoader  # noqa: E402


def _build_pseudonymizer(mode: str):
    if mode == "none":
        return None, None
    from app.ingestion.pii import (
        ConsistentPseudonymizer,
        InMemoryMappingStore,
        PostgresMappingStore,
        build_analyzer,
    )

    settings = get_settings()
    session = None
    if mode == "postgres":
        from app.persistence.database import SessionLocal

        session = SessionLocal()
        store = PostgresMappingStore(session)
    else:
        store = InMemoryMappingStore()
    pseudonymizer = ConsistentPseudonymizer(
        analyzer=build_analyzer(),
        mapping_store=store,
        salt=settings.PSEUDONYM_HASH_SALT,
        faker_locale=settings.PSEUDONYM_FAKER_LOCALE,
        language="es",
    )
    return pseudonymizer, session


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="presupuestos_json")
    parser.add_argument("--catalog", type=Path, default=ROOT / settings.CATALOG_PATH)
    parser.add_argument("--data-root", type=Path, default=ROOT / settings.INGESTION_DATA_ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / settings.PROCESSED_DATA_DIR)
    parser.add_argument(
        "--pseudonymize", choices=["none", "memory", "postgres"], default="none"
    )
    args = parser.parse_args(argv)

    catalog = load_catalog(args.catalog)
    records = load_budget_records(
        catalog, args.source, FileSystemLoader(data_root=args.data_root)
    )
    pseudonymizer, session = _build_pseudonymizer(args.pseudonymize)
    try:
        bundle = build_export(
            records,
            source_name=args.source,
            source_version=catalog.version,
            pseudonymizer=pseudonymizer,
        )
    finally:
        if session is not None:
            session.close()

    paths = write_export(bundle, args.out)
    report = bundle.report
    print(f"Ficheros leídos:      {report['files_read']}")
    print(f"Duplicados quitados:  {report['duplicates_removed']}")
    print(f"Válidos exportados:   {report['exported_clean']}")
    print(f"En cuarentena:        {report['exported_quarantine']}")
    print(f"Descartados:          {report['exported_discarded']}")
    for budget_id, info in report["rejections"].items():
        print(f"  - {budget_id}: {info['bucket']} ← {', '.join(info['reasons'])}")
    print("\nSalida:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
