"""Quick check of the storage layer. Run with:  python try_storage.py"""

import hashlib
import uuid

from app.schemas import OutcomeUpdate, ScanCreate
from app.storage.db import create_tables, make_engine, make_session_factory
from app.storage.object_store import InMemoryObjectStore, build_image_key
from app.storage.repository import InMemoryScanRepository, SqlScanRepository


def make_scan(success: bool, consent: bool = True) -> ScanCreate:
    return ScanCreate.model_validate({
        "scan_id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
        "captured_at": "2026-10-06T14:00:00+02:00",
        "device": {"platform": "ios", "device_model": "iPhone 15", "os_version": "17.5", "app_version": "4.12.0"},
        "camera": {"image_width": 1080, "image_height": 1920, "ambient_lux": 3.0},
        "location": {"latitude": 52.37, "longitude": 4.89, "accuracy_m": 8},
        "decode": {"success": success, "decoded_text": "NL*TNM*E12345*1" if success else None,
                   "decoder": "zxing" if success else "none", "duration_ms": 900},
        "consent_for_training": consent,
    })


def run(repo, name: str) -> None:
    images = InMemoryObjectStore()
    fake_image = b"\xff\xd8 pretend this is a jpeg"

    failed = make_scan(success=False)
    no_consent = make_scan(success=False, consent=False)

    for scan in (failed, no_consent):
        key = build_image_key(scan.scan_id, scan.captured_at)
        images.put_image(key, fake_image)
        assert repo.save_scan(scan, key, hashlib.sha256(fake_image).hexdigest())

    # The phone retries the same scan: must not be stored twice
    assert repo.save_scan(failed, "x", "y") is False

    # Driver typed the ID manually -> both scans get an outcome
    outcome = OutcomeUpdate(method="manual_entry", charger_id="CH-1",
                            evse_id="NL*TNM*E12345*1", resolved_at="2026-10-06T14:02:00+02:00")
    assert repo.save_outcome(failed.scan_id, outcome) is True
    assert repo.save_outcome(no_consent.scan_id, outcome) is False  # no consent -> not trainable

    labeled = repo.get_labeled_scans()
    assert len(labeled) == 1 and labeled[0].scan_id == failed.scan_id
    assert images.get_image(labeled[0].image_key) == fake_image
    print(f"{name}: OK -> {labeled[0].evse_id}, image {labeled[0].image_key}, lux {labeled[0].ambient_lux}")


engine = make_engine("sqlite:///:memory:")
create_tables(engine)
run(SqlScanRepository(make_session_factory(engine)), "SqlScanRepository")
run(InMemoryScanRepository(), "InMemoryScanRepository")
