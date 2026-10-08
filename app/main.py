"""
Entry point of the service.

Run locally:
    uvicorn app.main:app --reload
Then open http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.orm import sessionmaker

from app.api import predict, resolve, scans
from app.config import Settings
from app.inference.model import TextRecognizer, TrOCRRecognizer
from app.inference.pipeline import ScanPipeline
from app.storage.chargers import ChargerDirectory, InMemoryChargerDirectory, SqlChargerDirectory
from app.storage.db import create_tables, make_engine, make_session_factory
from app.storage.model_registry import InMemoryModelRegistry, ModelRegistry, SqlModelRegistry
from app.storage.object_store import InMemoryObjectStore, ObjectStore, S3ObjectStore
from app.storage.repository import InMemoryScanRepository, ScanRepository, SqlScanRepository

logger = logging.getLogger("qr_scan_service")


def build_session_factory(settings: Settings) -> sessionmaker | None:
    """One database connection pool, shared by the scan repository and the charger directory."""
    if not settings.database_url:
        logger.warning("DATABASE_URL not set: using in-memory storage (data is lost on restart)")
        return None
    engine = make_engine(settings.database_url)
    create_tables(engine)
    return make_session_factory(engine)


def build_charger_directory(settings: Settings, session_factory: sessionmaker | None) -> ChargerDirectory:
    if settings.chargers_csv:
        logger.info("Loading chargers from %s", settings.chargers_csv)
        return InMemoryChargerDirectory.from_csv(settings.chargers_csv)
    if session_factory:
        return SqlChargerDirectory(session_factory)
    logger.warning("No charger source configured: every charger lookup will come back empty")
    return InMemoryChargerDirectory()


def build_object_store(settings: Settings) -> ObjectStore:
    if settings.s3_bucket:
        return S3ObjectStore(settings.s3_bucket, endpoint_url=settings.s3_endpoint_url)
    logger.warning("S3_BUCKET not set: using in-memory image store (images are lost on restart)")
    return InMemoryObjectStore()


def create_app(
    settings: Settings | None = None,
    repository: ScanRepository | None = None,
    object_store: ObjectStore | None = None,
    recognizer: TextRecognizer | None = None,
    chargers: ChargerDirectory | None = None,
    registry: ModelRegistry | None = None,
) -> FastAPI:
    """Build the app. Tests pass in fake parts; production reads everything from env vars."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Runs once when the server starts (before the first request)
        needs_db = repository is None or chargers is None or registry is None
        session_factory = build_session_factory(settings) if needs_db else None

        app.state.settings = settings
        app.state.repository = repository or (
            SqlScanRepository(session_factory) if session_factory else InMemoryScanRepository()
        )
        app.state.chargers = chargers or build_charger_directory(settings, session_factory)
        app.state.object_store = object_store or build_object_store(settings)
        app.state.registry = registry or (
            SqlModelRegistry(session_factory) if session_factory else InMemoryModelRegistry()
        )

        app.state.recognizer = recognizer
        if recognizer is None and settings.load_model:
            app.state.recognizer = load_production_model(app.state.registry, settings)

        app.state.pipeline = ScanPipeline(app.state.chargers, app.state.recognizer)
        yield
        # Code after `yield` runs when the server shuts down (nothing to clean up yet)

    app = FastAPI(
        title="QR Scan Service",
        description="Collects charger-sticker scans and serves the image-to-text model.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(scans.router)
    app.include_router(resolve.router)
    app.include_router(predict.router)

    @app.get("/health", tags=["ops"])
    def health() -> dict:
        model = app.state.recognizer
        return {"status": "ok", "model_loaded": model is not None,
                "model_version": getattr(model, "version", None)}

    return app


def load_production_model(registry: ModelRegistry, settings: Settings) -> TrOCRRecognizer:
    """Load the model the registry marks as production (Step 12 put it
    there). If nothing has been promoted yet, use the configured base model.
    A rollback is just marking the previous version as production again and
    restarting the service."""
    production = registry.production()
    if production is not None:
        logger.info("Loading production model %s from %s", production.version, production.artifact_uri)
        return TrOCRRecognizer.from_pretrained(production.artifact_uri, version=production.version)
    logger.info("No production model in the registry; loading %s", settings.model_name)
    return TrOCRRecognizer.from_pretrained(settings.model_name)


app = create_app()
