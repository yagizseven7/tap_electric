"""
The golden test set: a FROZEN set of real scans every model is judged on.

Why frozen? If the test data changed between evaluations, a model could
look better simply because this month's test photos happen to be easier.
With one fixed set, the numbers of version 1 and version 7 are directly
comparable.

Rules that keep it honest:
  * Only stations that land in the "test" split (dataset.assign_split).
    Training only ever uses "train" and "val", and the split is stable,
    so no golden station can ever end up in training (no leakage).
  * Full photos, not crops. Validation runs the real pipeline (find the
    text line, read it, match it to nearby chargers), so a bad crop counts
    as a failure, exactly as it would for a driver. That also avoids the
    cropping bias of Step 10 (hard photos being dropped).
  * Images are COPIED into the golden folder. Normal scan images are
    deleted after the retention period (GDPR); golden images are kept, on
    the basis of the driver's training consent, and are reviewed by a person.
  * `verified` marks examples a human has checked. Labels from drivers can
    be wrong (wrong charger picked on the map); a person reviewing the
    golden set once removes that noise from the measurement.
  * Hard cases are kept on purpose: a test set of only easy photos would
    hide exactly the failures we care about.

A new golden set is created deliberately (e.g. once a year), never
silently, and it gets a new id. Reports always say which set they used.
"""

import hashlib
import io
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from app.inference.enhance import to_gray
from app.inference.matching import extract_evse_id
from app.storage.object_store import ObjectStore
from app.storage.repository import LabeledScan
from app.training.dataset import assign_split, conditions_of, station_key


@dataclass(frozen=True)
class GoldenExample:
    scan_id: str
    image_file: str             # relative to the golden folder
    evse_id: str                # what the sticker says
    charger_id: str             # which charger the driver actually used
    latitude: float | None
    longitude: float | None
    gps_accuracy_m: float | None
    hard: bool                  # the phone's QR scan failed
    conditions: list[str]       # dark / blurry / overexposed / low_contrast
    verified: bool = False      # checked by a person


@dataclass
class GoldenSet:
    directory: Path
    golden_id: str
    examples: list[GoldenExample]

    def image(self, example: GoldenExample) -> Image.Image:
        return Image.open(self.directory / example.image_file)


def build_golden_set(
    scans: list[LabeledScan],
    store: ObjectStore,
    out_dir: str | Path,
    val_percent: int = 10,
    test_percent: int = 10,
    max_examples: int | None = None,
) -> GoldenSet:
    """Freeze the test-split scans into a golden set. Refuses to overwrite an
    existing one: replacing the golden set must be a deliberate decision."""
    out = Path(out_dir)
    if (out / "golden.jsonl").exists():
        raise FileExistsError(f"A golden set already exists in {out}. Use load_golden_set, or pick a new folder.")
    (out / "images").mkdir(parents=True, exist_ok=True)

    examples = []
    for scan in sorted(scans, key=lambda s: str(s.scan_id)):
        label = extract_evse_id(scan.evse_id)
        if label is None or assign_split(station_key(label), val_percent, test_percent) != "test":
            continue
        try:
            data = store.get_image(scan.image_key)
            gray = to_gray(Image.open(io.BytesIO(data)))
        except (KeyError, OSError):
            continue
        image_file = f"images/{scan.scan_id}.jpg"
        (out / image_file).write_bytes(data)
        examples.append(GoldenExample(
            scan_id=str(scan.scan_id), image_file=image_file, evse_id=label, charger_id=scan.charger_id,
            latitude=scan.latitude, longitude=scan.longitude, gps_accuracy_m=scan.gps_accuracy_m,
            hard=not scan.decode_success, conditions=conditions_of(gray),
        ))
        if max_examples and len(examples) >= max_examples:
            break

    lines = [json.dumps(asdict(e), sort_keys=True) for e in examples]
    golden_id = "golden-" + hashlib.sha256("\n".join(lines).encode()).hexdigest()[:12]
    (out / "golden.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "golden_info.json").write_text(json.dumps({
        "golden_id": golden_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "examples": len(examples),
        "hard_examples": sum(e.hard for e in examples),
    }, indent=2), encoding="utf-8")
    return GoldenSet(out, golden_id, examples)


def load_golden_set(directory: str | Path) -> GoldenSet:
    directory = Path(directory)
    info = json.loads((directory / "golden_info.json").read_text(encoding="utf-8"))
    lines = (directory / "golden.jsonl").read_text(encoding="utf-8").splitlines()
    examples = [GoldenExample(**json.loads(line)) for line in lines if line.strip()]
    return GoldenSet(directory, info["golden_id"], examples)
