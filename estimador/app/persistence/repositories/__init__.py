"""Narrow repositories over the persistence layer.

Each repository wraps a single ``Session`` and exposes the operations its
caller actually needs — no ORM types leak past this boundary. Both
repositories are deliberately tiny: ingestion runs are infrequent and the
hot-path queries (mapping lookup) are point reads keyed by a composite index.
"""
from app.persistence.repositories.jobs import JobsRepository
from app.persistence.repositories.mappings import MappingsRepository

__all__ = ["JobsRepository", "MappingsRepository"]
