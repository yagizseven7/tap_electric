"""
Repositories: the only place that talks to the database.

The API and the training code call simple methods like save_scan() and
get_labeled_scans(). They never write SQL themselves. Benefits:
  * the rest of the code doesn't care which database is used
  * tests can use InMemoryScanRepository instead of a real database

Two implementations share the same methods:
  * SqlScanRepository      - PostgreSQL in production (works with SQLite in tests)
  * InMemoryScanRepository - plain Python dictionaries
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.schemas import OutcomeMethod, OutcomeUpdate, ScanCreate
from app.storage.db import ScanOutcomeRow, ScanRow


@dataclass(frozen=True)
class LabeledScan:
    """One training example: an image plus the correct answer."""
    scan_id: UUID
    image_key: str
    evse_id: str               # the target text the model should read
    charger_id: str
    method: str                # how the label was obtained (qr / manual_entry / map_selection)
    decode_success: bool       # True = phone could read it; False = a hard case
    captured_at: datetime
    device_model: str
    ambient_lux: float | None  # useful later to measure accuracy in dark conditions
    # Where the driver was: validation (Step 12) matches readings against nearby chargers
    latitude: float | None = None
    longitude: float | None = None
    gps_accuracy_m: float | None = None


def is_trainable(consent: bool, outcome: OutcomeUpdate) -> bool:
    """A scan becomes a training example only if the driver consented,
    the charger was found, and we know the text printed on the sticker."""
    return (
        consent
        and outcome.method != OutcomeMethod.ABANDONED
        and outcome.evse_id is not None
    )


class ScanRepository(Protocol):
    """The 'contract' every scan repository must follow."""

    def save_scan(self, scan: ScanCreate, image_key: str, image_sha256: str) -> bool: ...

    def scan_exists(self, scan_id: UUID) -> bool: ...

    def save_outcome(self, scan_id: UUID, outcome: OutcomeUpdate) -> bool: ...

    def get_labeled_scans(self, since: datetime | None = None, limit: int = 10_000) -> list[LabeledScan]: ...


# ---------------------------------------------------------------------------
# Real implementation (PostgreSQL via SQLAlchemy)
# ---------------------------------------------------------------------------

class SqlScanRepository:
    def __init__(self, session_factory: sessionmaker):
        self._session_factory = session_factory

    def save_scan(self, scan: ScanCreate, image_key: str, image_sha256: str) -> bool:
        """Store a scan. Returns False if this scan_id was already stored
        (the phone retried after a network error), so we don't store it twice."""
        with self._session_factory() as session:
            if session.get(ScanRow, scan.scan_id) is not None:
                return False

            loc = scan.location
            session.add(ScanRow(
                scan_id=scan.scan_id,
                session_id=scan.session_id,
                captured_at=scan.captured_at,
                platform=scan.device.platform.value,
                device_model=scan.device.device_model,
                os_version=scan.device.os_version,
                app_version=scan.device.app_version,
                camera=scan.camera.model_dump(),
                latitude=loc.latitude if loc else None,
                longitude=loc.longitude if loc else None,
                gps_accuracy_m=loc.accuracy_m if loc else None,
                decode_success=scan.decode.success,
                decoded_text=scan.decode.decoded_text,
                decoder=scan.decode.decoder.value,
                attempts=scan.decode.attempts,
                duration_ms=scan.decode.duration_ms,
                image_key=image_key,
                image_sha256=image_sha256,
                consent_for_training=scan.consent_for_training,
            ))
            session.commit()
            return True

    def scan_exists(self, scan_id: UUID) -> bool:
        with self._session_factory() as session:
            return session.get(ScanRow, scan_id) is not None

    def save_outcome(self, scan_id: UUID, outcome: OutcomeUpdate) -> bool:
        """Store (or overwrite) the outcome of a scan.
        Returns True if the scan can now be used for training."""
        with self._session_factory() as session:
            scan = session.get(ScanRow, scan_id)
            if scan is None:
                raise KeyError(f"Unknown scan_id {scan_id}")

            row = session.get(ScanOutcomeRow, scan_id) or ScanOutcomeRow(scan_id=scan_id)
            row.method = outcome.method.value
            row.charger_id = outcome.charger_id
            row.evse_id = outcome.evse_id
            row.resolved_at = outcome.resolved_at
            session.add(row)
            session.commit()
            return is_trainable(scan.consent_for_training, outcome)

    def get_labeled_scans(self, since: datetime | None = None, limit: int = 10_000) -> list[LabeledScan]:
        """All scans usable for training, newest first."""
        query = (
            select(ScanRow, ScanOutcomeRow)
            .join(ScanOutcomeRow, ScanOutcomeRow.scan_id == ScanRow.scan_id)
            .where(ScanRow.consent_for_training.is_(True))
            .where(ScanOutcomeRow.method != OutcomeMethod.ABANDONED.value)
            .where(ScanOutcomeRow.evse_id.is_not(None))
            .order_by(ScanRow.captured_at.desc())
            .limit(limit)
        )
        if since is not None:
            query = query.where(ScanRow.captured_at >= since)

        with self._session_factory() as session:
            return [_to_labeled(scan, outcome) for scan, outcome in session.execute(query)]


def _to_labeled(scan: ScanRow, outcome: ScanOutcomeRow) -> LabeledScan:
    return LabeledScan(
        scan_id=scan.scan_id,
        image_key=scan.image_key,
        evse_id=outcome.evse_id,
        charger_id=outcome.charger_id,
        method=outcome.method,
        decode_success=scan.decode_success,
        captured_at=scan.captured_at,
        device_model=scan.device_model,
        ambient_lux=(scan.camera or {}).get("ambient_lux"),
        latitude=scan.latitude,
        longitude=scan.longitude,
        gps_accuracy_m=scan.gps_accuracy_m,
    )


# ---------------------------------------------------------------------------
# Fake implementation for tests
# ---------------------------------------------------------------------------

class InMemoryScanRepository:
    def __init__(self) -> None:
        self._scans: dict[UUID, tuple[ScanCreate, str, str]] = {}
        self._outcomes: dict[UUID, OutcomeUpdate] = {}

    def save_scan(self, scan: ScanCreate, image_key: str, image_sha256: str) -> bool:
        if scan.scan_id in self._scans:
            return False
        self._scans[scan.scan_id] = (scan, image_key, image_sha256)
        return True

    def scan_exists(self, scan_id: UUID) -> bool:
        return scan_id in self._scans

    def save_outcome(self, scan_id: UUID, outcome: OutcomeUpdate) -> bool:
        if scan_id not in self._scans:
            raise KeyError(f"Unknown scan_id {scan_id}")
        self._outcomes[scan_id] = outcome
        return is_trainable(self._scans[scan_id][0].consent_for_training, outcome)

    def get_labeled_scans(self, since: datetime | None = None, limit: int = 10_000) -> list[LabeledScan]:
        result = []
        for scan_id, outcome in self._outcomes.items():
            scan, image_key, _ = self._scans[scan_id]
            if not is_trainable(scan.consent_for_training, outcome):
                continue
            if since is not None and scan.captured_at < since:
                continue
            result.append(LabeledScan(
                scan_id=scan_id,
                image_key=image_key,
                evse_id=outcome.evse_id,
                charger_id=outcome.charger_id,
                method=outcome.method.value,
                decode_success=scan.decode.success,
                captured_at=scan.captured_at,
                device_model=scan.device.device_model,
                ambient_lux=scan.camera.ambient_lux,
                latitude=scan.location.latitude if scan.location else None,
                longitude=scan.location.longitude if scan.location else None,
                gps_accuracy_m=scan.location.accuracy_m if scan.location else None,
            ))
        result.sort(key=lambda s: s.captured_at, reverse=True)
        return result[:limit]
