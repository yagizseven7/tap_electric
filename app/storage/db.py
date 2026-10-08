"""
PostgreSQL tables, defined with SQLAlchemy.

SQLAlchemy is an "ORM" (Object Relational Mapper): each Python class below
becomes a table, and each attribute becomes a column. We never write
CREATE TABLE statements by hand; SQLAlchemy generates them.

Only metadata is stored here. Images live in object storage (see
object_store.py); a scan row only keeps the image's key (its "path").
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Uuid,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Parent class of all tables."""


class ScanRow(Base):
    """One row per scan attempt sent by the app."""
    __tablename__ = "scans"

    # Identity
    scan_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    session_id: Mapped[uuid.UUID] = mapped_column(Uuid, index=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # Device (separate columns because we filter and group by them,
    # e.g. "failure rate per phone model")
    platform: Mapped[str] = mapped_column(String(20))
    device_model: Mapped[str] = mapped_column(String(100), index=True)
    os_version: Mapped[str] = mapped_column(String(50))
    app_version: Mapped[str] = mapped_column(String(50), index=True)

    # Camera settings: stored as JSON because the available fields differ per
    # phone and will grow over time. In PostgreSQL this becomes JSONB.
    camera: Mapped[dict] = mapped_column(JSON)

    # Location (nullable: the driver may deny location access).
    # In production you would add a PostGIS geography column for fast
    # "chargers within 50 m" queries.
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    gps_accuracy_m: Mapped[float | None] = mapped_column(Float)

    # What the phone decoded
    decode_success: Mapped[bool] = mapped_column(Boolean, index=True)
    decoded_text: Mapped[str | None] = mapped_column(String(2048))
    decoder: Mapped[str] = mapped_column(String(20))
    attempts: Mapped[int] = mapped_column(Integer)
    duration_ms: Mapped[int] = mapped_column(Integer)

    # Image reference + privacy
    image_key: Mapped[str] = mapped_column(String(500))
    image_sha256: Mapped[str] = mapped_column(String(64), index=True)
    consent_for_training: Mapped[bool] = mapped_column(Boolean)

    # One scan has at most one outcome
    outcome: Mapped["ScanOutcomeRow | None"] = relationship(back_populates="scan", uselist=False)


class ScanOutcomeRow(Base):
    """How the driver finally found (or did not find) the charger.

    This is what turns a scan into a labelled training example.
    Kept in a separate table because it arrives later, in a separate request.
    """
    __tablename__ = "scan_outcomes"

    scan_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("scans.scan_id"), primary_key=True)
    method: Mapped[str] = mapped_column(String(20), index=True)
    charger_id: Mapped[str | None] = mapped_column(String(100), index=True)
    evse_id: Mapped[str | None] = mapped_column(String(100))
    resolved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    scan: Mapped[ScanRow] = relationship(back_populates="outcome")


class ChargerRow(Base):
    """Known chargers. In reality this is owned by another Tap Electric service;
    here it is a read-only copy used to match a model prediction to a charger."""
    __tablename__ = "chargers"

    charger_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    evse_id: Mapped[str] = mapped_column(String(100), unique=True)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)


class ModelVersionRow(Base):
    """Which model versions exist and which one is in production.
    (MLflow keeps the full experiment history; this is a light mirror the
    API reads at startup to know which model to load.)"""
    __tablename__ = "model_versions"

    version: Mapped[str] = mapped_column(String(50), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    stage: Mapped[str] = mapped_column(String(20), default="candidate", index=True)  # candidate / production / archived / rejected
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    artifact_uri: Mapped[str] = mapped_column(String(500))
    # Lineage: lets us trace any model back to how it was made
    run_id: Mapped[str | None] = mapped_column(String(100))       # experiment-tracker run
    dataset_id: Mapped[str | None] = mapped_column(String(64))    # exact training data (Step 10)
    base_model: Mapped[str | None] = mapped_column(String(500))   # what it was fine-tuned from


def make_engine(database_url: str) -> Engine:
    """Create the connection to the database.

    Example URLs:
      postgresql+psycopg://user:password@localhost:5432/scans   (production)
      sqlite:///:memory:                                      (tests)
    """
    return create_engine(database_url)


def make_session_factory(engine: Engine) -> sessionmaker:
    """A session is one 'conversation' with the database; the factory makes new ones."""
    return sessionmaker(bind=engine, expire_on_commit=False)


def create_tables(engine: Engine) -> None:
    """Create all tables (in production you would use a migration tool like Alembic)."""
    Base.metadata.create_all(engine)
