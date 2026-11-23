# main.py
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app.routers import estimations, ingestion, sessions


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Session 6: fail fast on a malformed catalog rather than at the first
    ingestion request. A broken catalog is a deploy-time problem."""
    log = structlog.get_logger()
    try:
        from app.dependencies import get_catalog

        catalog = get_catalog()
        log.info(
            "catalog_loaded",
            version=catalog.version,
            sources_total=len(catalog.sources),
            sources_included=len(catalog.included_sources()),
        )
    except Exception as exc:  # noqa: BLE001
        log.error("catalog_load_failed", error=str(exc)[:400])
    yield


app = FastAPI(
    title="Estimador CAG",
    description="Sistema de estimación de software con arquitectura CAG",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(estimations.router)
app.include_router(sessions.router)
app.include_router(ingestion.router)


@app.get("/health")
async def health():
    return {"status": "healthy"}
