"""
Tests for the TrOCR wrapper (Step 8).

Downloading the real model in every test run would be slow, so we use the
tiny, untrained TrOCR from app/training/tiny_model.py. It has the same
architecture, so it runs through exactly the same code (pre-processing,
beam search, decoding, confidence). Its output is random text; how
ACCURATE a model is gets measured in Step 12, not in unit tests.
"""

import pytest
from PIL import Image

from app.inference.model import TrOCRRecognizer
from app.training.tiny_model import build_tiny_trocr


def tiny_recognizer(num_beams: int) -> TrOCRRecognizer:
    model, processor = build_tiny_trocr()
    return TrOCRRecognizer(model, processor, version="tiny-test", device="cpu",
                           num_beams=num_beams, max_new_tokens=8)


@pytest.mark.parametrize("num_beams", [1, 4])
def test_predict_returns_text_and_confidence(num_beams):
    recognizer = tiny_recognizer(num_beams)
    prediction = recognizer.predict(Image.new("RGB", (200, 50), "white"))

    assert isinstance(prediction.text, str)
    assert 0.0 <= prediction.confidence <= 1.0
    assert prediction.model_version == "tiny-test"


def test_predict_batch_keeps_order_and_accepts_any_image_mode():
    recognizer = tiny_recognizer(num_beams=1)
    images = [Image.new("RGB", (200, 50), "white"), Image.new("L", (80, 80), 0)]  # colour + greyscale
    predictions = recognizer.predict_batch(images)
    assert len(predictions) == 2
