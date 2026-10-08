"""
Tests for the scan pipeline (Step 9).

Sticker images are generated with OpenCV: a real QR code plus the printed
ID underneath, then damaged the way real photos are (darkness, sensor noise,
JPEG compression, blur). The model is replaced by a fake that returns a
chosen text, so we test the pipeline's logic, not the model's accuracy.
"""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.inference.enhance import find_text_lines
from app.inference.matching import extract_evse_id, similarity
from app.inference.pipeline import ResolveStatus, ScanPipeline
from app.inference.quality import measure_quality
from app.main import create_app
from app.storage.chargers import Charger, InMemoryChargerDirectory, SqlChargerDirectory, distance_m
from app.storage.db import ChargerRow, create_tables, make_engine, make_session_factory
from app.storage.object_store import InMemoryObjectStore
from app.storage.repository import InMemoryScanRepository
from tests.helpers import FakeRecognizer, destroyed_qr, photograph, sticker

AMSTERDAM = (52.3700, 4.8900)

CHARGER = Charger("CH-1", "NL*TNM*E12345*1", *AMSTERDAM)
TWIN = Charger("CH-2", "NL*TNM*E12345*2", *AMSTERDAM)        # second connector, same post
NEIGHBOUR = Charger("CH-3", "NL*TNM*E99871*1", 52.3702, 4.8903)  # ~30 m away
FAR_AWAY = Charger("CH-4", "NL*TNM*E12345*9", 52.3900, 4.8900)   # ~2.2 km away


def pipeline(recognizer=None, chargers=(CHARGER, TWIN, NEIGHBOUR, FAR_AWAY)) -> ScanPipeline:
    return ScanPipeline(InMemoryChargerDirectory(list(chargers)), recognizer)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "NL*TNM*E12345*1",
    "nl*tnm*e12345*1",
    "NLTNME12345*1",
    "https://example.com/start/NL%2ATNM%2AE12345%2A1",
    "ID: NL*TNM*E12345*1",
])
def test_extract_evse_id_handles_many_formats(text):
    assert extract_evse_id(text) == "NL*TNM*E12345*1"


def test_extract_evse_id_joins_an_id_split_by_spaces():
    # OCR sometimes puts spaces inside the ID and drops the asterisks
    assert similarity(extract_evse_id("NL TNM E12345 1"), "NL*TNM*E12345*1") == 1.0


def test_extract_evse_id_returns_none_without_an_id():
    assert extract_evse_id("hello world") is None


def test_similarity_tolerates_one_wrong_character():
    assert similarity("NL*TNM*E12345*1", "NLTNME12345*1") == 1.0  # asterisks ignored
    assert similarity("NL*TNM*E12345*1", "NL*TNM*E12346*1") > 0.9
    assert similarity("NL*TNM*E12345*1", "DE*ABC*E99999*3") < 0.5


# ---------------------------------------------------------------------------
# Charger lookup
# ---------------------------------------------------------------------------

def test_distance():
    assert distance_m(*AMSTERDAM, 52.3900, 4.8900) == pytest.approx(2224, rel=0.01)


@pytest.mark.parametrize("kind", ["memory", "sql"])
def test_find_near_and_get_by_evse(kind):
    if kind == "memory":
        directory = InMemoryChargerDirectory([CHARGER, NEIGHBOUR, FAR_AWAY])
    else:
        engine = make_engine("sqlite:///:memory:")
        create_tables(engine)
        factory = make_session_factory(engine)
        with factory() as session:
            session.add_all([ChargerRow(charger_id=c.charger_id, evse_id=c.evse_id,
                                        latitude=c.latitude, longitude=c.longitude)
                             for c in (CHARGER, NEIGHBOUR, FAR_AWAY)])
            session.commit()
        directory = SqlChargerDirectory(factory)

    near = {c.charger_id for c in directory.find_near(*AMSTERDAM, radius_m=150)}
    assert near == {"CH-1", "CH-3"}
    assert directory.get_by_evse("NL*TNM*E12345*1") == CHARGER
    assert directory.get_by_evse("NLTNME123451") == CHARGER  # asterisks are optional
    assert directory.get_by_evse("NL*TNM*E00000*1") is None


# ---------------------------------------------------------------------------
# Image quality
# ---------------------------------------------------------------------------

def test_quality_flags_dark_and_blurry_images():
    gray = lambda img: np.asarray(img.convert("L"))
    sharp = measure_quality(sticker())
    dark = measure_quality(gray(photograph(sticker(), brightness=0.08, noise=5)))
    blurry = measure_quality(gray(photograph(sticker(), blur=5, noise=5)))

    assert not (sharp.is_dark or sharp.is_blurry or sharp.is_overexposed)
    assert dark.is_dark and not dark.is_blurry  # dark but in focus: no blur hint
    assert blurry.is_blurry and not blurry.is_dark
    assert any("flashlight" in h for h in dark.hints())


def test_text_line_finder_crops_the_printed_id():
    crops = find_text_lines(np.asarray(destroyed_qr()))
    assert crops, "no text line found"
    height, width = crops[0].shape
    assert width > 4 * height  # the biggest crop is a wide, flat line of text


# ---------------------------------------------------------------------------
# The pipeline, stage by stage
# ---------------------------------------------------------------------------

def test_clean_qr_is_matched_without_the_model():
    recognizer = FakeRecognizer("should not be used")
    result = pipeline(recognizer).resolve(Image.fromarray(sticker()), *AMSTERDAM)

    assert result.status == ResolveStatus.MATCHED
    assert result.stage == "qr_raw"
    assert result.charger == CHARGER
    assert recognizer.calls == 0  # the cheap stage was enough
    assert result.hints == []


def test_readable_qr_for_an_unknown_charger_says_so():
    result = pipeline(chargers=()).resolve(Image.fromarray(sticker()), *AMSTERDAM)
    assert result.status == ResolveStatus.NOT_FOUND
    assert result.read_text == "NL*TNM*E12345*1"
    assert "don't recognise this charger" in result.hints[0]


@pytest.mark.parametrize("seed", range(5))
def test_very_dark_noisy_qr_is_rescued_by_enhancement(seed):
    photo = photograph(sticker(), brightness=0.05, noise=5, seed=seed)
    result = pipeline().resolve(photo, *AMSTERDAM)

    assert result.status == ResolveStatus.MATCHED
    assert result.stage.startswith("qr_") and result.stage != "qr_raw"


def test_unreadable_qr_falls_back_to_the_model():
    # The model misreads one character (5 -> 6). The pipeline still finds the
    # right charger because nothing else nearby looks similar.
    recognizer = FakeRecognizer("NL*TNM*E12346*1")
    result = pipeline(recognizer, chargers=(CHARGER, NEIGHBOUR)).resolve(destroyed_qr(), *AMSTERDAM)

    assert result.status == ResolveStatus.MATCHED
    assert result.stage == "ocr"
    assert result.charger == CHARGER
    assert result.model_version == "fake-1"


def test_two_similar_chargers_give_candidates_instead_of_a_guess():
    # Connectors *1 and *2 on the same post; the model read the last digit wrong.
    recognizer = FakeRecognizer("NL*TNM*E12345*7")
    result = pipeline(recognizer).resolve(destroyed_qr(), *AMSTERDAM)

    assert result.status == ResolveStatus.CANDIDATES
    assert {m.charger.charger_id for m in result.candidates[:2]} == {"CH-1", "CH-2"}
    assert result.hints  # the app gets something to tell the driver


def test_low_model_confidence_gives_candidates():
    recognizer = FakeRecognizer("NL*TNM*E12345*1", confidence=0.2)
    result = pipeline(recognizer, chargers=(CHARGER,)).resolve(destroyed_qr(), *AMSTERDAM)
    assert result.status == ResolveStatus.CANDIDATES


def test_chargers_outside_the_search_radius_are_ignored():
    recognizer = FakeRecognizer("NL*TNM*E12345*9")  # exactly FAR_AWAY's ID, but 2 km away
    result = pipeline(recognizer, chargers=(FAR_AWAY, NEIGHBOUR)).resolve(destroyed_qr(), *AMSTERDAM)
    assert result.charger is None


def test_without_location_only_an_exact_read_is_accepted():
    exact = pipeline(FakeRecognizer("NL*TNM*E12345*1")).resolve(destroyed_qr())
    almost = pipeline(FakeRecognizer("NL*TNM*E12346*1")).resolve(destroyed_qr())

    assert exact.status == ResolveStatus.MATCHED
    assert almost.status == ResolveStatus.NOT_FOUND


def test_without_a_model_an_unreadable_qr_is_not_found_with_hints():
    result = pipeline(recognizer=None).resolve(destroyed_qr(), *AMSTERDAM)
    assert result.status == ResolveStatus.NOT_FOUND
    assert result.stage == "none"
    assert result.hints


# ---------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------

def test_resolve_endpoint():
    app = create_app(
        settings=Settings(),
        repository=InMemoryScanRepository(),
        object_store=InMemoryObjectStore(),
        recognizer=FakeRecognizer("NL*TNM*E12346*1"),
        chargers=InMemoryChargerDirectory([CHARGER, NEIGHBOUR]),
    )
    _, png = cv2.imencode(".png", np.asarray(destroyed_qr()))
    with TestClient(app) as client:
        response = client.post(
            "/v1/resolve",
            files={"image": ("scan.png", png.tobytes(), "image/png")},
            data={"latitude": "52.37", "longitude": "4.89", "gps_accuracy_m": "10"},
        )
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "matched"
    assert body["stage"] == "ocr"
    assert body["charger_id"] == "CH-1"
    assert set(body["quality"]) >= {"brightness", "is_dark", "is_blurry"}
