"""Persistence layer for the Session 6 ingestion subsystem.

Holds the SQLAlchemy engine, the declarative ``Base`` and the row models that
back the pseudonymization mapping table and the ingestion job tracker. Higher
layers consume narrow repositories from ``app.persistence.repositories``; they
never see SQLAlchemy types directly.
"""
from app.persistence.database import (
    SessionLocal,
    create_engine_from_settings,
    get_session,
)
from app.persistence.models import Base, IngestionJobRow, PseudonymMappingRow

__all__ = [
    "Base",
    "IngestionJobRow",
    "PseudonymMappingRow",
    "SessionLocal",
    "create_engine_from_settings",
    "get_session",
]
