"""SQLAlchemy engine, session factory and per-request session helpers.

We use the *synchronous* SQLAlchemy 2.0 API. The ingestion paths in this module
are not on the hot user request path — they run as BackgroundTasks or one-shot
admin operations — so we trade async ergonomics for simplicity and less
moving infrastructure during teaching.
"""
from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


@lru_cache
def create_engine_from_settings() -> Engine:
    """Build the global engine from ``Settings.DATABASE_URL`` (singleton)."""
    return create_engine(
        get_settings().DATABASE_URL,
        pool_pre_ping=True,
        future=True,
    )


SessionLocal = sessionmaker(
    bind=create_engine_from_settings(),
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)


def get_session() -> Iterator[Session]:
    """FastAPI dependency that yields a Session and closes it on exit."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
