"""
Synthetic scans: fake but realistic sticker photos with known labels.

Two uses:
  1. Cold start: before the app has collected real data, synthetic stickers
     let us test and pre-train the training system.
  2. Demos and tests: run the whole system end to end on a laptop
     (try_training.py), which is what the assignment asks us to show.

Each fake scan goes through the real code: the image is stored with the
real repository, and whether the "phone" managed to decode it is decided
by actually running the QR decoder on it. So hard examples are hard for
the same reasons real ones are.

Real data always beats synthetic data. In production, synthetic examples
would only supplement real ones, and models are validated on real scans
only (Step 12).
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np

from app.inference.pipeline import decode_qr
from app.schemas import OutcomeUpdate, ScanCreate
from app.storage.chargers import Charger
from app.storage.object_store import ObjectStore, build_image_key
from app.storage.repository import ScanRepository

COUNTRIES = ["NL", "NL", "NL", "BE", "DE", "FR"]
OPERATORS = ["TNM", "ALF", "EVB", "NUO", "GFX", "ION", "LMS"]
FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX]

# How often each photo condition occurs. Roughly: most photos are fine.
CONDITIONS = {"good": 0.45, "dark": 0.15, "glare": 0.1, "blurry": 0.1, "faded": 0.1, "worn": 0.1}


def random_evse_id(rng: np.random.Generator) -> str:
    country = COUNTRIES[int(rng.integers(len(COUNTRIES)))]
    operator = OPERATORS[int(rng.integers(len(OPERATORS)))]
    number = "".join(str(d) for d in rng.integers(0, 10, int(rng.integers(4, 7))))
    return f"{country}*{operator}*E{number}*{int(rng.integers(1, 5))}"


def render_sticker(evse_id: str, rng: np.random.Generator) -> np.ndarray:
    """A sticker: QR code with the ID printed underneath, in a random font and size."""
    qr_size = int(rng.integers(220, 300))
    qr = cv2.resize(cv2.QRCodeEncoder.create().encode(evse_id), (qr_size, qr_size), interpolation=cv2.INTER_NEAREST)

    width = qr_size + int(rng.integers(120, 200))
    height = qr_size + int(rng.integers(160, 220))
    canvas = np.full((height, width), int(rng.integers(225, 256)), np.uint8)
    top = int(rng.integers(20, 50))
    left = (width - qr_size) // 2
    canvas[top: top + qr_size, left: left + qr_size] = qr

    font = FONTS[int(rng.integers(len(FONTS)))]
    thickness = int(rng.integers(2, 4))
    (text_width, _), _ = cv2.getTextSize(evse_id, font, 1.0, thickness)
    scale = rng.uniform(0.75, 0.9) * width / text_width       # fill most of the width
    (text_width, text_height), _ = cv2.getTextSize(evse_id, font, scale, thickness)
    x = (width - text_width) // 2
    y = top + qr_size + (height - top - qr_size + text_height) // 2
    cv2.putText(canvas, evse_id, (x, y), font, scale, int(rng.integers(0, 50)), thickness, cv2.LINE_AA)
    return canvas


def photograph(sticker: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    """Simulate a phone photo of the whole sticker under a given condition."""
    x = sticker.astype(np.float32)
    noise = 3.0
    if condition == "dark":
        x = x * rng.uniform(0.05, 0.2)
        noise = rng.uniform(4, 7)
    elif condition == "glare":
        x = x * rng.uniform(0.1, 0.25) + rng.uniform(200, 230)
    elif condition == "blurry":
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(2.5, 5.0))
    elif condition == "faded":
        alpha = rng.uniform(0.6, 0.8)
        x = x * (1 - alpha) + 215 * alpha
    elif condition == "worn":
        # A strip of the QR code is scratched away, the printed ID is intact
        top = int(np.argmax((sticker < 128).any(axis=1)))
        x[top: top + int(rng.integers(60, 140)), :] = rng.uniform(150, 220)
    x = np.clip(x + rng.normal(0, noise, x.shape), 0, 255).astype(np.uint8)
    _, encoded = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(55, 90))])
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


@dataclass
class SyntheticScan:
    scan: ScanCreate
    image: bytes
    outcome: OutcomeUpdate
    condition: str


def generate(
    n_chargers: int,
    scans_per_charger: int = 3,
    seed: int = 0,
    centre: tuple[float, float] = (52.37, 4.89),
) -> tuple[list[Charger], list[SyntheticScan]]:
    rng = np.random.default_rng(seed)
    names, weights = list(CONDITIONS), np.array(list(CONDITIONS.values()))
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)

    # Chargers come in STATIONS of 1-4 connectors at the same spot, whose IDs
    # differ only in the last digit (NL*TNM*E12345*1, *2, ...). That's how
    # real charging points look, and it's the hard case for matching: a
    # misread last digit points at the neighbouring connector.
    chargers, used = [], set()
    while len(chargers) < n_chargers:
        station = random_evse_id(rng).rsplit("*", 1)[0]
        if station in used:
            continue
        used.add(station)
        lat = centre[0] + rng.uniform(-0.02, 0.02)
        lon = centre[1] + rng.uniform(-0.03, 0.03)
        for connector in range(1, int(rng.integers(1, 5)) + 1):
            if len(chargers) == n_chargers:
                break
            chargers.append(Charger(f"CH-{len(chargers) + 1:05d}", f"{station}*{connector}",
                                    lat + rng.uniform(-2e-5, 2e-5), lon + rng.uniform(-2e-5, 2e-5)))

    scans = []
    for charger in chargers:
        sticker = render_sticker(charger.evse_id, rng)  # one physical sticker per charger
        for _ in range(scans_per_charger):
            condition = str(rng.choice(names, p=weights / weights.sum()))
            photo = photograph(sticker, condition, rng)
            decoded = decode_qr(photo)
            success = decoded is not None
            captured = start + timedelta(minutes=int(rng.integers(0, 60 * 24 * 30)))
            gps_error = rng.normal(0, 8, 2)  # metres north, metres east

            scan = ScanCreate.model_validate({
                "scan_id": str(uuid.UUID(bytes=rng.bytes(16), version=4)),
                "session_id": str(uuid.UUID(bytes=rng.bytes(16), version=4)),
                "captured_at": captured.isoformat(),
                "device": {"platform": "android", "device_model": "Synthetic", "os_version": "15", "app_version": "4.12.0"},
                "camera": {"image_width": photo.shape[1], "image_height": photo.shape[0],
                           "ambient_lux": 3.0 if condition == "dark" else 400.0},
                # Phone GPS is off by roughly 5-20 m: enough to find the station,
                # not enough to tell its connectors (1-2 m apart) apart
                "location": {"latitude": charger.latitude + gps_error[0] / 111_000,
                             "longitude": charger.longitude + gps_error[1] / 68_000,
                             "accuracy_m": round(float(np.hypot(*gps_error)) + 5, 1)},
                "decode": {"success": success, "decoded_text": decoded, "decoder": "opencv" if success else "none",
                           "duration_ms": 800 if success else 6000},
                "consent_for_training": True,
            })
            outcome = OutcomeUpdate(
                method="qr" if success else str(rng.choice(["manual_entry", "map_selection"])),
                charger_id=charger.charger_id,
                evse_id=charger.evse_id,
                resolved_at=(captured + timedelta(seconds=40)).isoformat(),
            )
            _, jpg = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, 95])
            scans.append(SyntheticScan(scan, jpg.tobytes(), outcome, condition))
    return chargers, scans


def store_scans(scans: list[SyntheticScan], repo: ScanRepository, store: ObjectStore) -> None:
    """Push synthetic scans through the same storage code the API uses."""
    for s in scans:
        key = build_image_key(s.scan.scan_id, s.scan.captured_at)
        store.put_image(key, s.image)
        repo.save_scan(s.scan, image_key=key, image_sha256="synthetic")
        repo.save_outcome(s.scan.scan_id, s.outcome)
