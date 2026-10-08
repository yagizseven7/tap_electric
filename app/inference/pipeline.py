"""
The scan pipeline: from a camera image to a charger.

    image
      │
      ├─ 1. decode the QR code as-is                  (fast, free)
      ├─ 2. enhance the image, decode again            (fast, free)
      ├─ 3. read the printed ID with the model (TrOCR) (slower)
      │     and compare it with chargers near the driver
      ▼
    MATCHED     -> start the session with this charger
    CANDIDATES  -> show the driver 2-3 options ("Is it one of these?")
    NOT_FOUND   -> show tips and the manual entry field

Every stage that fails hands over to the next, so we only pay for the
model when the cheap methods have failed. The driver's choice in the
CANDIDATES case comes back through PATCH /outcome and becomes a training
label: the pipeline helps collect exactly the hard examples the model
needs to learn from.
"""

import time
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np
from PIL import Image

from app.inference.enhance import enhanced_variants, find_text_lines, prepare_for_model, to_gray
from app.inference.matching import MatchCandidate, extract_evse_id, rank_chargers
from app.inference.model import TextRecognizer
from app.inference.quality import ImageQuality, measure_quality
from app.storage.chargers import Charger, ChargerDirectory


class ResolveStatus(str, Enum):
    MATCHED = "matched"
    CANDIDATES = "candidates"
    NOT_FOUND = "not_found"


@dataclass
class PipelineResult:
    status: ResolveStatus
    stage: str                      # which stage produced the answer, e.g. "qr_raw", "qr_clahe", "ocr"
    quality: ImageQuality
    charger: Charger | None = None
    candidates: list[MatchCandidate] = field(default_factory=list)
    read_text: str | None = None    # what the QR decoder or model read
    ocr_confidence: float | None = None
    model_version: str | None = None
    duration_ms: float = 0.0

    @property
    def hints(self) -> list[str]:
        if self.status == ResolveStatus.MATCHED:
            return []
        if self.stage.startswith("qr_"):
            # The image was fine: we read the code, we just don't know this charger
            return ["We read the code but don't recognise this charger. Please type the ID printed on the sticker."]
        if self.status == ResolveStatus.CANDIDATES:
            return ["Is it one of these chargers?"] + self.quality.hints()
        return self.quality.hints() or ["Move closer so the sticker fills the frame."]


def decode_qr(gray: np.ndarray) -> str | None:
    """Try to read a QR code. Returns its text, or None.

    OpenCV's decoder is used here because it's easy to install. zxing-cpp
    is usually more robust and is a drop-in replacement for this function.
    """
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(gray)
    return text or None


class ScanPipeline:
    def __init__(
        self,
        chargers: ChargerDirectory,
        recognizer: TextRecognizer | None = None,
        min_match_score: float = 0.85,     # how similar the read ID must be to a charger's ID
        min_margin: float = 0.08,          # how much better the best charger must be than the 2nd
        min_ocr_confidence: float = 0.5,   # how sure the model must be
        min_candidate_score: float = 0.5,  # below this, a charger isn't even worth suggesting
        max_candidates: int = 3,
    ):
        self.chargers = chargers
        self.recognizer = recognizer
        self.min_match_score = min_match_score
        self.min_margin = min_margin
        self.min_ocr_confidence = min_ocr_confidence
        self.min_candidate_score = min_candidate_score
        self.max_candidates = max_candidates

    def resolve(
        self,
        image: Image.Image,
        latitude: float | None = None,
        longitude: float | None = None,
        gps_accuracy_m: float | None = None,
        use_qr: bool = True,
    ) -> PipelineResult:
        """use_qr=False skips stages 1 and 2. Production never does that;
        evaluation (Step 12) does, to measure the model on EVERY photo
        instead of only the few where the QR code failed."""
        start = time.perf_counter()
        result = self._resolve(image, latitude, longitude, gps_accuracy_m, use_qr)
        result.duration_ms = (time.perf_counter() - start) * 1000
        return result

    @staticmethod
    def try_qr(gray: np.ndarray) -> tuple[str, str] | None:
        """Stages 1 and 2: returns (decoded text, stage name), or None."""
        payload = decode_qr(gray)                       # Stage 1: as-is
        if payload:
            return payload, "qr_raw"
        for name, variant in enhanced_variants(gray):   # Stage 2: each enhancement, first success wins
            payload = decode_qr(variant)
            if payload:
                return payload, f"qr_{name}"
        return None

    # ------------------------------------------------------------------

    def _resolve(self, image, latitude, longitude, gps_accuracy_m, use_qr=True) -> PipelineResult:
        gray = to_gray(image)
        quality = measure_quality(gray)

        if use_qr:
            decoded = self.try_qr(gray)
            if decoded:
                return self._from_qr(*decoded, quality)

        # Stage 3: read the printed ID with the model
        if self.recognizer is None:
            return PipelineResult(ResolveStatus.NOT_FOUND, "none", quality)
        return self._from_ocr(gray, quality, latitude, longitude, gps_accuracy_m)

    def _from_qr(self, payload: str, stage: str, quality: ImageQuality) -> PipelineResult:
        """A decoded QR code is reliable, so we look the ID up exactly."""
        evse_id = extract_evse_id(payload)
        charger = self.chargers.get_by_evse(evse_id) if evse_id else None
        # If the QR holds something other than an EVSE ID (e.g. an operator's
        # own URL), production would call Tap Electric's existing lookup here.
        status = ResolveStatus.MATCHED if charger else ResolveStatus.NOT_FOUND
        return PipelineResult(status, stage, quality, charger=charger, read_text=payload)

    def _from_ocr(self, gray, quality, latitude, longitude, gps_accuracy_m) -> PipelineResult:
        # The model reads one line at a time: crop likely text lines first.
        # If none are found, give it the whole image as a last resort.
        # prepare_for_model is the same function training uses (Step 10).
        crops = find_text_lines(gray) or [gray]
        predictions = self.recognizer.predict_batch([prepare_for_model(c) for c in crops])
        best = max(predictions, key=lambda p: p.confidence)
        base = dict(stage="ocr", quality=quality, model_version=best.model_version)

        # Without a location we can't search nearby, so only an exact ID will do
        if latitude is None or longitude is None:
            for p in sorted(predictions, key=lambda p: p.confidence, reverse=True):
                evse_id = extract_evse_id(p.text)
                charger = self.chargers.get_by_evse(evse_id) if evse_id else None
                if charger and p.confidence >= self.min_ocr_confidence:
                    return PipelineResult(ResolveStatus.MATCHED, charger=charger, read_text=p.text,
                                          ocr_confidence=p.confidence, **base)
            return PipelineResult(ResolveStatus.NOT_FOUND, read_text=best.text,
                                  ocr_confidence=best.confidence, **base)

        # With a location: compare every reading with every nearby charger
        nearby = self.chargers.find_near(latitude, longitude, self._search_radius(gps_accuracy_m))
        scores: dict[str, MatchCandidate] = {}
        source = {}  # charger_id -> the prediction that matched it best
        for p in predictions:
            for m in rank_chargers(p.text, nearby):
                if m.charger.charger_id not in scores or m.score > scores[m.charger.charger_id].score:
                    scores[m.charger.charger_id] = m
                    source[m.charger.charger_id] = p
        ranked = sorted(scores.values(), key=lambda m: m.score, reverse=True)

        if not ranked or ranked[0].score < self.min_candidate_score:
            return PipelineResult(ResolveStatus.NOT_FOUND, read_text=best.text,
                                  ocr_confidence=best.confidence, **base)

        top = ranked[0]
        top_prediction = source[top.charger.charger_id]
        runner_up = ranked[1].score if len(ranked) > 1 else 0.0
        confident = (
            top.score >= self.min_match_score
            and top.score - runner_up >= self.min_margin
            and top_prediction.confidence >= self.min_ocr_confidence
        )
        if confident:
            return PipelineResult(ResolveStatus.MATCHED, charger=top.charger, candidates=[top],
                                  read_text=top_prediction.text, ocr_confidence=top_prediction.confidence, **base)

        suggestions = [m for m in ranked if m.score >= self.min_candidate_score][: self.max_candidates]
        return PipelineResult(ResolveStatus.CANDIDATES, candidates=suggestions,
                              read_text=top_prediction.text, ocr_confidence=top_prediction.confidence, **base)

    @staticmethod
    def _search_radius(gps_accuracy_m: float | None) -> float:
        """Search wider when the GPS is less precise, but within 150 m - 1 km."""
        return min(max(150.0, 3 * (gps_accuracy_m or 50.0)), 1000.0)
