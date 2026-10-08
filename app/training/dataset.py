"""
Step 10: from labelled scans to a training dataset.

    repository.get_labeled_scans()            (Step 6)
          │
          ├─ 1. split      train / val / test, by charger STATION, stable over time
          ├─ 2. crop       cut out the line with the printed ID (the model reads one line)
          ├─ 3. measure    dark? blurry? glare? (Step 12 reports accuracy per condition)
          └─ 4. manifest   one JSON line per example + a dataset id (a fingerprint)
          │
          ▼
    OcrDataset (PyTorch)  loads a crop, damages it (train only), prepares it
                          exactly like the live pipeline, and tokenises the label

The built dataset is a folder:

    data/<version>/
        crops/<scan_id>.png     the cropped text lines (original, undamaged)
        manifest.jsonl          one example per line
        dataset_info.json       id, counts, what was dropped and why
"""

import hashlib
import io
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from app.inference.enhance import character_counts, find_text_lines, prepare_for_model, to_gray
from app.inference.matching import extract_evse_id, normalize, similarity
from app.inference.model import TextRecognizer
from app.inference.quality import measure_quality
from app.storage.object_store import ObjectStore
from app.storage.repository import LabeledScan
from app.training.augment import Augmenter

# ---------------------------------------------------------------------------
# 1. Splitting
# ---------------------------------------------------------------------------

def station_key(evse_id: str) -> str:
    """The charging station an EVSE belongs to: drop the connector number.

    NL*TNM*E12345*1 and NL*TNM*E12345*2 are two connectors on the same post.
    Their stickers look nearly identical and are photographed in the same
    spot, so they must land in the same split.
    """
    found = extract_evse_id(evse_id) or normalize(evse_id)
    parts = found.split("*")
    return "*".join(parts[:3]) if len(parts) > 3 else found


def assign_split(group: str, val_percent: int = 10, test_percent: int = 10) -> str:
    """Put every station in train, val or test, based on a hash of its key.

    Why not a random split?
      * Leakage: if one photo of a sticker is in train and another photo of
        the SAME sticker is in test, the model is tested on something it has
        practically seen. Its score would look better than it is. So all
        scans of one station go to the same split.
      * Stability: a hash gives the same answer every time. When we retrain
        next month with more data, a station that was in test stays in
        test, so old test data never leaks into training and results stay
        comparable between model versions.
    """
    bucket = int(hashlib.sha256(group.encode()).hexdigest(), 16) % 100
    if bucket < test_percent:
        return "test"
    if bucket < test_percent + val_percent:
        return "val"
    return "train"


# ---------------------------------------------------------------------------
# 2. Cropping: where on the photo is the printed ID?
# ---------------------------------------------------------------------------

class Cropper(Protocol):
    def crop(self, gray: np.ndarray, label: str) -> np.ndarray | None: ...


class CharacterCountCropper:
    """Pick the text line whose number of characters matches the label.

    No model needed. The label NL*TNM*E12345*1 has 11 letters and digits,
    so we look for a line with about 11 character-shaped blobs. A strip of
    a damaged QR code or an empty patch of background fails this check, and
    if no line passes, the example is dropped rather than trained on.

    Used when no trained model is available yet (the very first training
    run, the demo). Once a model exists, LabelGuidedCropper is stricter.
    """

    def __init__(self, tolerance: float = 0.3):
        self.tolerance = tolerance  # allow 30% difference: blur merges letters, dirt adds blobs

    def crop(self, gray: np.ndarray, label: str) -> np.ndarray | None:
        expected = sum(ch.isalnum() for ch in label)
        allowed = max(2, round(self.tolerance * expected))
        best, best_error = None, None
        for crop in find_text_lines(gray):
            error = min(abs(count - expected) for count in character_counts(crop))
            if error <= allowed and (best_error is None or error < best_error):
                best, best_error = crop, error
        return best


class LabelGuidedCropper:
    """Take the text line that the current model reads closest to the label.

    For every labelled photo we know WHAT the sticker says but not WHERE.
    A sticker often has several lines of text (operator name, phone number,
    the ID), so we ask the current model to read each line and keep the one
    most similar to the known ID. If nothing is similar enough, the example
    is dropped. This also filters out wrong labels: if no line resembles
    the label, the driver may have picked the wrong charger on the map.

    Known weakness: on the hardest photos the current model may read every
    line badly, so exactly those examples are more likely to be dropped. A
    trained text detector (or a small hand-annotated set of boxes) removes
    this bias; it's listed as future work.
    """

    def __init__(self, recognizer: TextRecognizer, min_similarity: float = 0.5):
        self.recognizer = recognizer
        self.min_similarity = min_similarity

    def crop(self, gray: np.ndarray, label: str) -> np.ndarray | None:
        crops = find_text_lines(gray)
        if not crops:
            return None
        predictions = self.recognizer.predict_batch([prepare_for_model(c) for c in crops])
        scores = [similarity(extract_evse_id(p.text) or p.text, label) for p in predictions]
        best = int(np.argmax(scores))
        return crops[best] if scores[best] >= self.min_similarity else None


# ---------------------------------------------------------------------------
# 3 + 4. Building the dataset
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ManifestRow:
    scan_id: str
    split: str
    text: str                 # the label: what the model should read
    crop_file: str            # relative to the dataset folder
    crop_sha256: str
    station: str
    charger_id: str
    hard: bool                # the phone failed to decode this scan
    label_method: str         # qr / manual_entry / map_selection
    conditions: list[str]     # e.g. ["dark", "blurry"]; used for slice metrics in Step 12
    captured_at: str


@dataclass
class DatasetBuild:
    directory: Path
    dataset_id: str
    rows: list[ManifestRow]
    dropped: dict[str, int] = field(default_factory=dict)
    # How many scans of each photo condition were kept and dropped. If dark
    # photos are dropped much more often than clear ones, the dataset is
    # losing exactly the hard cases; this makes that visible.
    kept_by_condition: dict[str, int] = field(default_factory=dict)
    dropped_by_condition: dict[str, int] = field(default_factory=dict)

    def split(self, name: str) -> list[ManifestRow]:
        return [r for r in self.rows if r.split == name]

    def counts(self) -> dict[str, int]:
        return dict(Counter(r.split for r in self.rows))


def conditions_of(gray: np.ndarray) -> list[str]:
    q = measure_quality(gray)
    flags = {"dark": q.is_dark, "overexposed": q.is_overexposed,
             "low_contrast": q.is_low_contrast, "blurry": q.is_blurry}
    return [name for name, on in flags.items() if on]


def build_dataset(
    scans: list[LabeledScan],
    store: ObjectStore,
    cropper: Cropper,
    out_dir: str | Path,
    val_percent: int = 10,
    test_percent: int = 10,
) -> DatasetBuild:
    out = Path(out_dir)
    (out / "crops").mkdir(parents=True, exist_ok=True)

    rows: list[ManifestRow] = []
    dropped: Counter = Counter()
    kept_by_condition: Counter = Counter()
    dropped_by_condition: Counter = Counter()

    for scan in sorted(scans, key=lambda s: str(s.scan_id)):  # sorted -> same order every run
        label = extract_evse_id(scan.evse_id)
        if label is None:
            dropped["label_not_an_evse_id"] += 1
            continue

        try:
            gray = to_gray(Image.open(io.BytesIO(store.get_image(scan.image_key))))
        except (KeyError, OSError):
            dropped["image_missing_or_broken"] += 1
            continue

        conditions = conditions_of(gray)
        crop = cropper.crop(gray, label)
        if crop is None:
            dropped["no_matching_text_line"] += 1
            dropped_by_condition.update(conditions or ["clear"])
            continue
        kept_by_condition.update(conditions or ["clear"])

        png = _png_bytes(crop)
        crop_file = f"crops/{scan.scan_id}.png"
        (out / crop_file).write_bytes(png)

        station = station_key(label)
        rows.append(ManifestRow(
            scan_id=str(scan.scan_id),
            split=assign_split(station, val_percent, test_percent),
            text=label,
            crop_file=crop_file,
            crop_sha256=hashlib.sha256(png).hexdigest(),
            station=station,
            charger_id=scan.charger_id,
            hard=not scan.decode_success,
            label_method=scan.method,
            conditions=conditions,
            captured_at=_iso(scan.captured_at),
        ))

    manifest_lines = [json.dumps(asdict(r), sort_keys=True) for r in rows]
    (out / "manifest.jsonl").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    # A fingerprint of the exact data: same scans + same crops -> same id.
    # Logged with every training run, so any model can be traced back to
    # the precise data it was trained on (reproducibility).
    dataset_id = hashlib.sha256("\n".join(manifest_lines).encode()).hexdigest()[:16]

    build = DatasetBuild(out, dataset_id, rows, dict(dropped),
                         dict(kept_by_condition), dict(dropped_by_condition))
    (out / "dataset_info.json").write_text(json.dumps({
        "dataset_id": dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "examples": len(rows),
        "splits": build.counts(),
        "hard_examples": sum(r.hard for r in rows),
        "dropped": build.dropped,
        "kept_by_condition": build.kept_by_condition,
        "dropped_by_condition": build.dropped_by_condition,
    }, indent=2), encoding="utf-8")
    return build


def load_manifest(directory: str | Path) -> list[ManifestRow]:
    lines = (Path(directory) / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    return [ManifestRow(**json.loads(line)) for line in lines if line.strip()]


def _png_bytes(gray: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(gray).save(buffer, format="PNG")  # PNG is lossless: no extra damage
    return buffer.getvalue()


def _iso(value: datetime) -> str:
    if value.tzinfo is None:  # SQLite returns times without a timezone
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


# ---------------------------------------------------------------------------
# PyTorch dataset
# ---------------------------------------------------------------------------

class OcrDataset(Dataset):
    """Serves (image, label) pairs to the training loop.

    For each example:
      1. load the original crop (greyscale)
      2. TRAIN ONLY: damage it randomly (augment.py)
      3. prepare it exactly like the live pipeline (prepare_for_model)
      4. the processor resizes and normalises it into numbers (pixel_values)
      5. the tokenizer turns the label text into token numbers (labels)
    """

    def __init__(self, rows: list[ManifestRow], root: str | Path, processor,
                 augmenter: Augmenter | None = None, max_target_length: int = 32):
        self.rows = rows
        self.root = Path(root)
        self.processor = processor
        self.augmenter = augmenter
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        gray = np.asarray(Image.open(self.root / row.crop_file).convert("L"))
        if self.augmenter is not None:
            gray = self.augmenter(gray)
        pixel_values = self.processor(images=prepare_for_model(gray), return_tensors="pt").pixel_values[0]
        labels = self.processor.tokenizer(
            row.text, max_length=self.max_target_length, truncation=True
        ).input_ids
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels), "text": row.text}


def collate_batch(batch: list[dict]) -> dict:
    """Combine examples into one batch. Labels have different lengths, so
    shorter ones are padded with -100: PyTorch's loss function ignores -100,
    so the padding doesn't count as something the model should predict."""
    longest = max(len(item["labels"]) for item in batch)
    labels = torch.full((len(batch), longest), -100, dtype=torch.long)
    for i, item in enumerate(batch):
        labels[i, : len(item["labels"])] = item["labels"]
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": labels,
        "texts": [item["text"] for item in batch],
    }


def sample_weights(rows: list[ManifestRow], hard_weight: float = 3.0) -> list[float]:
    """How often each example is drawn during training.

    Failed scans are rare but they are exactly the cases the model exists
    for, so they are drawn `hard_weight` times as often ("oversampling").
    """
    return [hard_weight if (r.hard or r.conditions) else 1.0 for r in rows]
