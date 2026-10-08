"""
Contract tests for the storage layer (Step 6).

Most tests use the fast in-memory fakes. That is only safe if the fakes
behave exactly like the real implementations. So every test here runs
TWICE: once against the in-memory version and once against the real SQL
version (on a temporary SQLite database). If both pass the same tests,
they keep the same promises ("contract").
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.schemas import OutcomeUpdate, ScanCreate
from app.storage.db import create_tables, make_engine, make_session_factory
from app.storage.object_store import InMemoryObjectStore, S3ObjectStore, build_image_key
from app.storage.repository import InMemoryScanRepository, SqlScanRepository


@pytest.fixture(params=["memory", "sql"])
def repo(request):
    if request.param == "memory":
        yield InMemoryScanRepository()
        return
    engine = make_engine("sqlite:///:memory:")
    create_tables(engine)
    yield SqlScanRepository(make_session_factory(engine))
    engine.dispose()


def scan(captured_at="2026-10-06T14:00:00+02:00", consent=True, location=True) -> ScanCreate:
    return ScanCreate.model_validate({
        "scan_id": str(uuid.uuid4()), "session_id": str(uuid.uuid4()), "captured_at": captured_at,
        "device": {"platform": "ios", "device_model": "iPhone 15", "os_version": "17.5", "app_version": "4.12.0"},
        "camera": {"image_width": 1080, "image_height": 1920, "ambient_lux": 3.0},
        "location": {"latitude": 52.37, "longitude": 4.89, "accuracy_m": 8} if location else None,
        "decode": {"success": False, "duration_ms": 900},
        "consent_for_training": consent,
    })


def outcome(method="manual_entry", charger_id="CH-1", evse_id="NL*TNM*E12345*1") -> OutcomeUpdate:
    return OutcomeUpdate(method=method, charger_id=charger_id, evse_id=evse_id,
                         resolved_at="2026-10-06T14:05:00+02:00")


def test_save_scan_once_then_report_duplicate(repo):
    s = scan()
    assert repo.save_scan(s, "k", "sha") is True
    assert repo.save_scan(s, "k", "sha") is False
    assert repo.scan_exists(s.scan_id)
    assert not repo.scan_exists(uuid.uuid4())


def test_outcome_for_unknown_scan_raises(repo):
    with pytest.raises(KeyError):
        repo.save_outcome(uuid.uuid4(), outcome())


@pytest.mark.parametrize("consent,result,trainable", [
    (True, outcome(), True),
    (False, outcome(), False),                                                   # no consent
    (True, OutcomeUpdate(method="abandoned", resolved_at="2026-10-06T14:05:00+02:00"), False),  # no charger
    (True, outcome(evse_id=None), False),                                        # no text to learn
])
def test_only_consented_resolved_scans_become_training_data(repo, consent, result, trainable):
    s = scan(consent=consent)
    repo.save_scan(s, "k", "sha")
    assert repo.save_outcome(s.scan_id, result) is trainable
    assert len(repo.get_labeled_scans()) == (1 if trainable else 0)


def test_labeled_scans_carry_everything_training_needs(repo):
    s = scan()
    repo.save_scan(s, "scans/a.jpg", "sha")
    repo.save_outcome(s.scan_id, outcome())
    [labeled] = repo.get_labeled_scans()
    assert (labeled.scan_id, labeled.image_key, labeled.evse_id, labeled.charger_id) == \
        (s.scan_id, "scans/a.jpg", "NL*TNM*E12345*1", "CH-1")
    assert labeled.decode_success is False and labeled.method == "manual_entry"
    assert (labeled.latitude, labeled.longitude, labeled.gps_accuracy_m) == (52.37, 4.89, 8)
    assert labeled.ambient_lux == 3.0


def test_scan_without_location_is_still_usable(repo):
    s = scan(location=False)
    repo.save_scan(s, "k", "sha")
    repo.save_outcome(s.scan_id, outcome())
    assert repo.get_labeled_scans()[0].latitude is None


def test_newest_first_with_since_and_limit(repo):
    times = ["2026-10-01T10:00:00+00:00", "2026-10-03T10:00:00+00:00", "2026-10-05T10:00:00+00:00"]
    for t in times:
        s = scan(captured_at=t)
        repo.save_scan(s, "k", "sha")
        repo.save_outcome(s.scan_id, outcome())

    def day(labeled):
        return labeled.captured_at.day

    assert [day(x) for x in repo.get_labeled_scans()] == [5, 3, 1]
    assert [day(x) for x in repo.get_labeled_scans(limit=2)] == [5, 3]
    since = datetime(2026, 10, 2, tzinfo=timezone.utc)
    assert [day(x) for x in repo.get_labeled_scans(since=since)] == [5, 3]


def test_outcome_can_be_corrected(repo):
    s = scan()
    repo.save_scan(s, "k", "sha")
    repo.save_outcome(s.scan_id, outcome(charger_id="CH-1"))
    repo.save_outcome(s.scan_id, outcome(charger_id="CH-2", evse_id="NL*TNM*E12345*2"))  # driver changed charger
    [labeled] = repo.get_labeled_scans()
    assert labeled.charger_id == "CH-2"


# ---------------------------------------------------------------------------
# Image storage
# ---------------------------------------------------------------------------

def test_image_key_uses_date_folders():
    key = build_image_key(uuid.UUID(int=1), datetime(2026, 10, 6, tzinfo=timezone.utc))
    assert key == "scans/2026/10/06/00000000-0000-0000-0000-000000000001.jpg"


def test_in_memory_store_round_trip_and_missing_key():
    store = InMemoryObjectStore()
    store.put_image("a.jpg", b"123")
    assert store.get_image("a.jpg") == b"123"
    with pytest.raises(KeyError):
        store.get_image("missing.jpg")


def test_s3_store_encrypts_and_reads_back():
    # boto3 is replaced by a mock: we check WHAT would be sent to S3, without S3
    client = MagicMock()
    client.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=b"jpeg"))}
    with patch("boto3.client", return_value=client):
        store = S3ObjectStore("qr-scans", endpoint_url="http://minio:9000")
    store.put_image("scans/a.jpg", b"jpeg")
    sent = client.put_object.call_args.kwargs
    assert sent["Bucket"] == "qr-scans" and sent["Key"] == "scans/a.jpg"
    assert sent["ServerSideEncryption"] == "AES256"
    assert store.get_image("scans/a.jpg") == b"jpeg"


# ---------------------------------------------------------------------------
# Experiment tracking
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_mlflow_tracker_records_a_run(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    from app.training.tracking import MlflowTracker

    monkeypatch.chdir(tmp_path)  # MLflow stores artifact files relative to the working folder
    tracker = MlflowTracker(f"sqlite:///{tmp_path}/mlflow.db", experiment="test")
    run_id = tracker.start("v-test", {"learning_rate": 1e-3, "dataset_id": "abc"})
    tracker.log_metrics({"val_cer": 0.5, "bad": float("nan")}, step=1)  # NaN is skipped, not an error
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text("{}")
    tracker.log_artifacts(str(tmp_path / "model"))
    tracker.end()

    run = mlflow.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["dataset_id"] == "abc"
    assert run.data.metrics == {"val_cer": 0.5}
