"""
End-to-end test: the whole feedback loop, through the real HTTP API.

    phone app ──POST /v1/scans──▶ ┐
              ──PATCH /outcome──▶ ┘ stored + labelled        (Steps 5-7)
                                     │
                     training job ◀──┘ dataset + training    (Steps 10-11)
                                     │
                    promotion gate ◀─┘ golden set evaluation (Step 12)
                                     │
    restarted API loads the promoted model from the registry
    phone app ──POST /v1/resolve──▶ answered by the NEW model (Step 9)

Every other test checks one part. This one checks that the parts fit
together: the data the API stores is the data the trainer can read, the
model the trainer saves is one the API can load, and so on. It uses the
tiny model with lenient quality bars, because it tests the plumbing, not
how well the model reads.
"""

import json
from dataclasses import asdict

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.storage.chargers import InMemoryChargerDirectory
from app.storage.model_registry import InMemoryModelRegistry
from app.storage.object_store import InMemoryObjectStore
from app.storage.repository import InMemoryScanRepository
from app.training.dataset import CharacterCountCropper
from app.training.golden import build_golden_set
from app.training.job import RetrainPolicy, Validation, run_training_job
from app.training.promotion import PromotionPolicy
from app.training.synthetic import generate
from app.training.tiny_model import build_tiny_trocr
from app.training.tracking import InMemoryTracker
from app.training.train import TrainConfig
from tests.helpers import destroyed_qr, png_bytes

# Lets any model through: this test is about the plumbing, not model quality
OPEN_GATE = PromotionPolicy(min_exact_match=0.0, min_resolved_rate=0.0, min_gain_over_gps=-1.0,
                            max_wrong_match_rate=1.0, max_p95_latency_ms=1e9)


@pytest.mark.slow
def test_scans_in_trained_model_out(tmp_path):
    chargers, scans = generate(n_chargers=60, scans_per_charger=2, seed=21)
    repo, store, registry = InMemoryScanRepository(), InMemoryObjectStore(), InMemoryModelRegistry()
    directory = InMemoryChargerDirectory(chargers)
    shared = dict(repository=repo, object_store=store, chargers=directory, registry=registry)

    # 1. The app sends every scan and, later, its outcome
    with TestClient(create_app(settings=Settings(), **shared)) as api:
        for s in scans:
            created = api.post("/v1/scans", data={"metadata": s.scan.model_dump_json()},
                               files={"image": ("scan.jpg", s.image, "image/jpeg")})
            assert created.status_code == 201, created.text
            outcome = api.patch(f"/v1/scans/{s.scan.scan_id}/outcome",
                                json=json.loads(s.outcome.model_dump_json()))
            assert outcome.json()["labelled"] is True
        assert api.get("/health").json()["model_loaded"] is False  # nothing trained yet

    # 2. Freeze a golden set, then 3. train, validate and promote
    golden = build_golden_set(repo.get_labeled_scans(), store, tmp_path / "golden", test_percent=25)
    assert golden.examples, "the golden set needs at least one station"
    result = run_training_job(
        repo=repo, store=store, registry=registry, tracker=InMemoryTracker(),
        config=TrainConfig(base_model="tiny", output_dir=str(tmp_path / "models"), epochs=1, batch_size=16,
                           learning_rate=1e-3, min_improvement=-1, device="cpu"),
        policy=RetrainPolicy(min_train_examples=10),
        validation=Validation(golden, directory, OPEN_GATE, reports_dir=str(tmp_path / "reports")),
        data_dir=str(tmp_path / "data"), load_model=lambda name: build_tiny_trocr(),
        cropper=CharacterCountCropper(),
    )
    assert result.status == "promoted", result.reason
    assert registry.production().version == result.version
    assert (tmp_path / "reports" / f"{result.version}.json").exists()

    # 4. A restarted API loads the promoted model and uses it for failed scans
    charger = chargers[0]
    with TestClient(create_app(settings=Settings(load_model=True), **shared)) as api:
        assert api.get("/health").json()["model_version"] == result.version
        answer = api.post("/v1/resolve",
                          files={"image": ("worn.png", png_bytes(destroyed_qr(charger.evse_id)), "image/png")},
                          data={"latitude": str(charger.latitude), "longitude": str(charger.longitude)})
    body = answer.json()
    assert answer.status_code == 200, body
    assert body["stage"] == "ocr"                    # the QR code was unreadable, so the model answered
    assert body["model_version"] == result.version   # ...and it was the newly promoted model
    assert body["status"] in {"matched", "candidates", "not_found"}


def test_synthetic_scans_are_valid_api_payloads():
    """Fast guard for the slow test above: the generated scans must pass the
    same schema validation the API applies."""
    _, scans = generate(n_chargers=3, scans_per_charger=1, seed=1)
    for s in scans:
        assert json.loads(s.scan.model_dump_json())["scan_id"] == str(s.scan.scan_id)
        assert asdict(s)["condition"] in {"good", "dark", "glare", "blurry", "faded", "worn"}
