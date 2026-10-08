"""
Tests for the API endpoints. Run with:  pytest

They use the in-memory storage and a fake model, so no database, S3 or
model download is needed.
"""

import io
import json
import uuid

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app
from app.storage.object_store import InMemoryObjectStore
from app.storage.repository import InMemoryScanRepository
from tests.helpers import FakeRecognizer


@pytest.fixture
def parts():
    return {"repo": InMemoryScanRepository(), "store": InMemoryObjectStore()}


@pytest.fixture
def client(parts):
    app = create_app(
        settings=Settings(max_image_bytes=1_000_000),
        repository=parts["repo"],
        object_store=parts["store"],
        recognizer=FakeRecognizer(confidence=0.93),
    )
    with TestClient(app) as c:  # "with" runs the startup code (lifespan)
        yield c


def jpeg_with_exif() -> bytes:
    """A small test image that carries EXIF data (camera make), like a real photo."""
    img = Image.new("RGB", (120, 60), "white")
    exif = Image.Exif()
    exif[0x010F] = "SecretPhoneMaker"  # 0x010F = "Make" tag
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def scan_metadata(**overrides) -> dict:
    data = {
        "scan_id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
        "captured_at": "2026-10-06T14:00:00+02:00",
        "device": {"platform": "android", "device_model": "Pixel 8", "os_version": "15", "app_version": "4.12.0"},
        "camera": {"image_width": 120, "image_height": 60, "ambient_lux": 4.0},
        "location": {"latitude": 52.37, "longitude": 4.89, "accuracy_m": 10},
        "decode": {"success": False, "duration_ms": 5000, "attempts": 40},
        "consent_for_training": True,
    }
    data.update(overrides)
    return data


def post_scan(client, metadata: dict, image: bytes | None = None, content_type: str = "image/jpeg"):
    return client.post(
        "/v1/scans",
        data={"metadata": json.dumps(metadata)},
        files={"image": ("scan.jpg", image if image is not None else jpeg_with_exif(), content_type)},
    )


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True, "model_version": "fake-1"}


def test_create_scan_stores_image_without_exif(client, parts):
    meta = scan_metadata()
    response = post_scan(client, meta)

    assert response.status_code == 201
    assert response.json()["status"] == "stored"
    assert parts["repo"].scan_exists(uuid.UUID(meta["scan_id"]))

    key = f"scans/2026/10/06/{meta['scan_id']}.jpg"
    stored = Image.open(io.BytesIO(parts["store"].get_image(key)))
    assert len(stored.getexif()) == 0  # EXIF removed


def test_retry_is_not_stored_twice(client):
    meta = scan_metadata()
    assert post_scan(client, meta).status_code == 201
    second = post_scan(client, meta)
    assert second.status_code == 200
    assert second.json()["status"] == "duplicate"


def test_invalid_metadata_is_rejected(client):
    meta = scan_metadata(location={"latitude": 500, "longitude": 4.89})
    response = post_scan(client, meta)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["location", "latitude"]


def test_wrong_file_type_is_rejected(client):
    response = post_scan(client, scan_metadata(), image=b"hello", content_type="text/plain")
    assert response.status_code == 415


def test_broken_image_is_rejected(client):
    response = post_scan(client, scan_metadata(), image=b"not really a jpeg")
    assert response.status_code == 422


def test_too_large_image_is_rejected(client):
    response = post_scan(client, scan_metadata(), image=b"\xff" * 1_000_001)
    assert response.status_code == 413


def test_outcome_makes_scan_trainable(client, parts):
    meta = scan_metadata()
    post_scan(client, meta)

    response = client.patch(
        f"/v1/scans/{meta['scan_id']}/outcome",
        json={"method": "manual_entry", "charger_id": "CH-1", "evse_id": "NL*TNM*E12345*1",
              "resolved_at": "2026-10-06T14:02:00+02:00"},
    )
    assert response.status_code == 200
    assert response.json()["labelled"] is True
    assert len(parts["repo"].get_labeled_scans()) == 1


def test_outcome_for_unknown_scan_returns_404(client):
    response = client.patch(
        f"/v1/scans/{uuid.uuid4()}/outcome",
        json={"method": "abandoned", "resolved_at": "2026-10-06T14:02:00+02:00"},
    )
    assert response.status_code == 404


def test_predict(client):
    response = client.post("/v1/predict", files={"image": ("s.jpg", jpeg_with_exif(), "image/jpeg")})
    assert response.status_code == 200
    assert response.json() == {"text": "NL*TNM*E12345*1", "confidence": 0.93, "model_version": "fake-1"}


def test_predict_without_model_returns_503(parts):
    app = create_app(settings=Settings(), repository=parts["repo"], object_store=parts["store"])
    with TestClient(app) as c:
        response = c.post("/v1/predict", files={"image": ("s.jpg", jpeg_with_exif(), "image/jpeg")})
    assert response.status_code == 503
