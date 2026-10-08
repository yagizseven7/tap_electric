"""
Model registry: which model versions exist and which one is in production.

Life of a model version:

    candidate  ──(passes validation, Step 12)──▶  production  ──(replaced)──▶  archived
        │
        └──(fails validation)──▶  rejected

Training (Step 11) only ever creates CANDIDATES. Promoting a candidate to
production is a separate decision with its own checks (Step 12), so a bad
training run can never reach drivers by itself.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.storage.db import ModelVersionRow

CANDIDATE, PRODUCTION, ARCHIVED, REJECTED = "candidate", "production", "archived", "rejected"


@dataclass
class ModelVersion:
    version: str
    artifact_uri: str                 # where the model files are (local folder or s3://...)
    stage: str = CANDIDATE
    metrics: dict = field(default_factory=dict)
    run_id: str | None = None         # the experiment-tracker run that produced it
    dataset_id: str | None = None     # the exact data it was trained on
    base_model: str | None = None     # what it was fine-tuned from
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ModelRegistry(Protocol):
    def register(self, model: ModelVersion) -> None: ...

    def get(self, version: str) -> ModelVersion | None: ...

    def production(self) -> ModelVersion | None: ...

    def latest(self) -> ModelVersion | None: ...

    def set_stage(self, version: str, stage: str) -> None: ...


class InMemoryModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, ModelVersion] = {}

    def register(self, model: ModelVersion) -> None:
        self._models[model.version] = model

    def get(self, version: str) -> ModelVersion | None:
        return self._models.get(version)

    def production(self) -> ModelVersion | None:
        return next((m for m in self._models.values() if m.stage == PRODUCTION), None)

    def latest(self) -> ModelVersion | None:
        return max(self._models.values(), key=lambda m: m.created_at, default=None)

    def set_stage(self, version: str, stage: str) -> None:
        if stage == PRODUCTION:  # only one production model at a time
            for m in self._models.values():
                if m.stage == PRODUCTION:
                    m.stage = ARCHIVED
        self._models[version].stage = stage


class SqlModelRegistry:
    def __init__(self, session_factory: sessionmaker):
        self._session_factory = session_factory

    def register(self, model: ModelVersion) -> None:
        with self._session_factory() as session:
            session.add(ModelVersionRow(
                version=model.version, created_at=model.created_at, stage=model.stage,
                metrics=model.metrics, artifact_uri=model.artifact_uri, run_id=model.run_id,
                dataset_id=model.dataset_id, base_model=model.base_model,
            ))
            session.commit()

    def get(self, version: str) -> ModelVersion | None:
        with self._session_factory() as session:
            row = session.get(ModelVersionRow, version)
            return _to_model(row) if row else None

    def production(self) -> ModelVersion | None:
        return self._first(select(ModelVersionRow).where(ModelVersionRow.stage == PRODUCTION))

    def latest(self) -> ModelVersion | None:
        return self._first(select(ModelVersionRow).order_by(ModelVersionRow.created_at.desc()))

    def set_stage(self, version: str, stage: str) -> None:
        with self._session_factory() as session:
            if stage == PRODUCTION:
                for row in session.scalars(select(ModelVersionRow).where(ModelVersionRow.stage == PRODUCTION)):
                    row.stage = ARCHIVED
            session.get(ModelVersionRow, version).stage = stage
            session.commit()  # one transaction: there is never a moment with two production models

    def _first(self, query) -> ModelVersion | None:
        with self._session_factory() as session:
            row = session.scalars(query.limit(1)).first()
            return _to_model(row) if row else None


def _to_model(row: ModelVersionRow) -> ModelVersion:
    created = row.created_at if row.created_at.tzinfo else row.created_at.replace(tzinfo=timezone.utc)
    return ModelVersion(
        version=row.version, artifact_uri=row.artifact_uri, stage=row.stage, metrics=dict(row.metrics or {}),
        run_id=row.run_id, dataset_id=row.dataset_id, base_model=row.base_model, created_at=created,
    )
