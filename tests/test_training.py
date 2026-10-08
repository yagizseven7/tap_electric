"""
Tests for Step 10 (dataset) and Step 11 (training).

Everything runs on small synthetic data and the tiny model, so the whole
file takes well under a minute and needs no downloads.
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from app.inference.model import TrOCRRecognizer
from app.storage.db import create_tables, make_engine, make_session_factory
from app.storage.model_registry import (
    ARCHIVED, CANDIDATE, PRODUCTION, InMemoryModelRegistry, ModelVersion, SqlModelRegistry,
)
from app.storage.object_store import InMemoryObjectStore
from app.storage.repository import InMemoryScanRepository, LabeledScan
from app.training import augment
from app.training.augment import STAGES, Augmenter
from app.training.dataset import (
    CharacterCountCropper, OcrDataset, assign_split, build_dataset, collate_batch, load_manifest,
    sample_weights, station_key,
)
from app.training.job import RetrainPolicy, run_training_job
from app.training.metrics import cer, edit_distance, exact_match
from app.training.synthetic import generate, render_sticker, store_scans
from app.training.tiny_model import build_tiny_trocr
from app.training.tracking import InMemoryTracker
from app.training.train import TrainConfig, train

# ---------------------------------------------------------------------------
# Shared synthetic data (built once for the whole file)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stored_scans():
    _, scans = generate(n_chargers=80, scans_per_charger=2, seed=3)  # ~30 stations
    repo, store = InMemoryScanRepository(), InMemoryObjectStore()
    store_scans(scans, repo, store)
    return repo, store


@pytest.fixture(scope="module")
def dataset(stored_scans, tmp_path_factory):
    repo, store = stored_scans
    return build_dataset(repo.get_labeled_scans(), store, CharacterCountCropper(),
                         tmp_path_factory.mktemp("data"), val_percent=20, test_percent=20)


def fast_config(tmp_path: Path, **overrides) -> TrainConfig:
    settings = dict(base_model="tiny", output_dir=str(tmp_path / "models"), epochs=2, batch_size=8,
                    learning_rate=1e-3, device="cpu", eval_batch_size=16)
    settings.update(overrides)
    return TrainConfig(**settings)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def crop_image() -> np.ndarray:
    img = np.full((40, 300), 230, np.uint8)
    cv2.putText(img, "NL*TNM*E12345*1", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 20, 2)
    return img


@pytest.mark.parametrize("fn", [fn for _, options in STAGES for fn in options], ids=lambda f: f.__name__)
def test_every_degradation_keeps_shape_and_type(fn):
    out = fn(crop_image(), np.random.default_rng(0))
    assert out.shape == (40, 300) and out.dtype == np.uint8


def test_darken_makes_darker_and_overexpose_brighter():
    img, rng = crop_image(), np.random.default_rng(0)
    assert augment.darken(img, rng).mean() < img.mean() * 0.5
    assert augment.overexpose(img, rng).std() < img.std() * 0.6  # contrast washed out


def test_augmenter_is_repeatable_and_p0_changes_nothing():
    img = crop_image()
    assert np.array_equal(Augmenter(p=1.0, seed=5)(img), Augmenter(p=1.0, seed=5)(img))
    assert np.array_equal(Augmenter(p=0.0, seed=5)(img), img)
    assert not np.array_equal(Augmenter(p=1.0, seed=5)(img), img)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_edit_distance():
    assert edit_distance("E12345", "E12345") == 0
    assert edit_distance("E12345", "E12845") == 1   # replace
    assert edit_distance("E12345", "E1234") == 1    # delete
    assert edit_distance("", "ABC") == 3


def test_cer_and_exact_match_ignore_asterisks_and_case():
    targets = ["NL*TNM*E12345*1", "NL*TNM*E99999*2"]
    assert exact_match(["nltnme123451", "NL*TNM*E99999*3"], targets) == 0.5
    assert cer(["NL*TNM*E12345*1", "NL*TNM*E99999*2"], targets) == 0.0
    assert cer(["NL*TNM*E12845*1", "NL*TNM*E99999*2"], targets) == pytest.approx(1 / 24)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def test_connectors_of_one_station_share_a_split():
    assert station_key("NL*TNM*E12345*1") == station_key("NL*TNM*E12345*2") == "NL*TNM*E12345"
    assert assign_split(station_key("NL*TNM*E12345*1")) == assign_split(station_key("NL*TNM*E12345*2"))


def test_split_is_stable_and_roughly_80_10_10():
    keys = [f"NL*TNM*E{i:05d}" for i in range(3000)]
    first = [assign_split(k) for k in keys]
    assert first == [assign_split(k) for k in keys]  # same answer every time
    share = {s: first.count(s) / len(first) for s in ("train", "val", "test")}
    assert 0.75 < share["train"] < 0.85 and 0.07 < share["val"] < 0.13 and 0.07 < share["test"] < 0.13


# ---------------------------------------------------------------------------
# Building the dataset
# ---------------------------------------------------------------------------

def test_dataset_is_built_without_leakage(dataset):
    assert len(dataset.rows) >= 50
    assert set(dataset.counts()) == {"train", "val", "test"}

    splits_per_station = defaultdict(set)
    for row in dataset.rows:
        splits_per_station[row.station].add(row.split)
        assert (dataset.directory / row.crop_file).exists()
    assert all(len(s) == 1 for s in splits_per_station.values())  # no station in two splits

    assert load_manifest(dataset.directory) == dataset.rows
    assert (dataset.directory / "dataset_info.json").exists()


def test_dataset_id_is_a_fingerprint(stored_scans, dataset, tmp_path):
    repo, store = stored_scans
    scans = repo.get_labeled_scans()
    again = build_dataset(scans, store, CharacterCountCropper(), tmp_path / "a", val_percent=20, test_percent=20)
    fewer = build_dataset(scans[:-5], store, CharacterCountCropper(), tmp_path / "b", val_percent=20, test_percent=20)
    assert again.dataset_id == dataset.dataset_id   # same data -> same id
    assert fewer.dataset_id != dataset.dataset_id   # different data -> different id


def test_cropper_rejects_a_sticker_without_readable_text():
    sticker = render_sticker("NL*TNM*E12345*1", np.random.default_rng(0))
    sticker[sticker.shape[0] * 2 // 3:, :] = 240  # wipe the printed line, keep the QR code
    assert CharacterCountCropper().crop(sticker, "NL*TNM*E12345*1") is None


def test_missing_images_are_counted_not_crashed(tmp_path):
    scan = LabeledScan(scan_id="00000000-0000-4000-8000-000000000000", image_key="nowhere.jpg",
                       evse_id="NL*TNM*E12345*1", charger_id="CH-1", method="qr", decode_success=True,
                       captured_at=datetime.now(timezone.utc), device_model="x", ambient_lux=None)
    build = build_dataset([scan], InMemoryObjectStore(), CharacterCountCropper(), tmp_path)
    assert build.rows == [] and build.dropped == {"image_missing_or_broken": 1}


def test_torch_dataset_and_collate(dataset):
    _, processor = build_tiny_trocr()
    rows = dataset.split("train")[:4]
    ds = OcrDataset(rows, dataset.directory, processor, augmenter=Augmenter(p=1.0, seed=0))
    batch = collate_batch([ds[i] for i in range(4)])

    assert batch["pixel_values"].shape == (4, 3, 32, 128)
    assert batch["texts"] == [r.text for r in rows]
    lengths = [len(r.text) + 2 for r in rows]  # characters + <s> + </s>
    for i, length in enumerate(lengths):
        assert (batch["labels"][i, :length] != -100).all()
        assert (batch["labels"][i, length:] == -100).all()  # padding is ignored by the loss


def test_hard_examples_get_more_weight(dataset):
    weights = sample_weights(dataset.rows, hard_weight=3.0)
    for row, weight in zip(dataset.rows, weights):
        assert weight == (3.0 if row.hard or row.conditions else 1.0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def datasets_for(dataset, processor):
    return (OcrDataset(dataset.split("train"), dataset.directory, processor, augmenter=Augmenter(seed=0)),
            OcrDataset(dataset.split("val"), dataset.directory, processor))


@pytest.mark.slow
def test_training_saves_a_model_that_loads_and_predicts(dataset, tmp_path):
    model, processor = build_tiny_trocr()
    train_ds, val_ds = datasets_for(dataset, processor)
    tracker = InMemoryTracker()
    # min_improvement=-1 counts every epoch as an improvement, so the save
    # path always runs (2 epochs are too few to guarantee real progress).
    result = train(fast_config(tmp_path, min_improvement=-1), model, processor, train_ds, val_ds, tracker, "v-test")

    # (An untrained model can score a CER above 100% by inventing extra
    # characters, so we only require that some epoch was saved.)
    assert result.improved and result.best_epoch >= 1
    assert [m["step"] for m in tracker.runs[0]["metrics"]] == [0, 1, 2]  # baseline + 2 epochs
    assert tracker.runs[0]["status"] == "FINISHED"
    assert np.isfinite([h["train_loss"] for h in result.history[1:]]).all()

    reloaded = TrOCRRecognizer.from_pretrained(result.model_dir, version="v-test", device="cpu", num_beams=1)
    prediction = reloaded.predict(Image.new("RGB", (300, 40), "white"))
    assert prediction.model_version == "v-test"


@pytest.mark.slow
def test_no_improvement_means_nothing_is_saved(dataset, tmp_path):
    model, processor = build_tiny_trocr()
    train_ds, val_ds = datasets_for(dataset, processor)
    # learning rate 0: the weights never change, so validation can't improve
    result = train(fast_config(tmp_path, epochs=1, learning_rate=0.0), model, processor,
                   train_ds, val_ds, InMemoryTracker(), "v-flat")
    assert not result.improved and result.model_dir is None
    assert not (tmp_path / "models" / "v-flat").exists()


# ---------------------------------------------------------------------------
# Retrain policy, registry and the full job
# ---------------------------------------------------------------------------

def labeled(at: datetime, success: bool) -> LabeledScan:
    return LabeledScan("00000000-0000-4000-8000-000000000000", "k", "NL*TNM*E1*1", "CH", "qr", success, at, "x", None)


def test_retrain_policy():
    policy = RetrainPolicy(min_new_examples=3, min_new_hard_examples=2)
    last = ModelVersion("v1", "models/v1", created_at=datetime(2026, 10, 1, tzinfo=timezone.utc))
    after = last.created_at + timedelta(days=1)
    before = last.created_at - timedelta(days=1)

    assert policy.decide([], None)[0]  # no model yet
    assert not policy.decide([labeled(after, True), labeled(before, True)], last)[0]
    assert policy.decide([labeled(after, True)] * 3, last)[0]    # enough new scans
    assert policy.decide([labeled(after, False)] * 2, last)[0]   # enough new failed scans
    # SQLite gives times without a timezone; they must still compare
    assert policy.decide([labeled(after.replace(tzinfo=None), True)] * 3, last)[0]


@pytest.mark.parametrize("kind", ["memory", "sql"])
def test_registry_keeps_one_production_model(kind):
    if kind == "memory":
        registry = InMemoryModelRegistry()
    else:
        engine = make_engine("sqlite:///:memory:")
        create_tables(engine)
        registry = SqlModelRegistry(make_session_factory(engine))

    registry.register(ModelVersion("v1", "models/v1", created_at=datetime(2026, 10, 1, tzinfo=timezone.utc)))
    registry.register(ModelVersion("v2", "models/v2", dataset_id="abc",
                                   created_at=datetime(2026, 10, 5, tzinfo=timezone.utc)))
    assert registry.production() is None
    assert registry.latest().version == "v2"
    assert registry.get("v2").dataset_id == "abc"

    registry.set_stage("v1", PRODUCTION)
    registry.set_stage("v2", PRODUCTION)
    assert registry.production().version == "v2"
    assert registry.get("v1").stage == ARCHIVED


@pytest.mark.slow
def test_training_job_end_to_end(stored_scans, tmp_path):
    repo, store = stored_scans
    registry = InMemoryModelRegistry()
    common = dict(repo=repo, store=store, registry=registry, tracker=InMemoryTracker(),
                  data_dir=str(tmp_path / "data"), load_model=lambda name: build_tiny_trocr(),
                  cropper=CharacterCountCropper(), policy=RetrainPolicy(min_train_examples=10))

    first = run_training_job(config=fast_config(tmp_path, epochs=1, min_improvement=-1), **common)
    assert first.status == "registered"
    candidate = registry.get(first.version)
    assert candidate.stage == CANDIDATE
    assert candidate.dataset_id == first.dataset_id
    assert Path(candidate.artifact_uri).exists()

    # Nothing new since that model -> the policy skips the next run
    second = run_training_job(config=fast_config(tmp_path), **common)
    assert second.status == "skipped"
