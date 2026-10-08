"""
Shared test helpers: a fake model and generated sticker images.

Kept in one place so every test file uses the same fakes. (Pytest
fixtures live in conftest.py; plain helper functions and classes live here
and are imported normally: `from tests.helpers import FakeRecognizer`.)
"""

import cv2
import numpy as np
from PIL import Image

from app.inference.model import Prediction


class FakeRecognizer:
    """Stands in for TrOCR: 'reads' the same text from every image.

    Lets us test the API, pipeline and evaluation logic in milliseconds,
    with a predictable answer, without loading a real model.
    """

    def __init__(self, text: str = "NL*TNM*E12345*1", confidence: float = 0.9, version: str = "fake-1"):
        self.text, self.confidence, self.version = text, confidence, version
        self.calls = 0

    def predict(self, image):
        return self.predict_batch([image])[0]

    def predict_batch(self, images):
        self.calls += 1
        return [Prediction(self.text, self.confidence, self.version) for _ in images]


def sticker(text: str = "NL*TNM*E12345*1") -> np.ndarray:
    """A white sticker with a QR code and the ID printed underneath."""
    qr = cv2.QRCodeEncoder.create().encode(text)
    qr = cv2.resize(qr, (300, 300), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((520, 460), 255, np.uint8)
    canvas[40:340, 80:380] = qr
    cv2.putText(canvas, text, (40, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 3)
    return canvas


def photograph(img: np.ndarray, brightness: float = 1.0, offset: float = 0, noise: float = 0,
               blur: float = 0, seed: int = 0) -> Image.Image:
    """Simulate a phone photo: exposure, blur, sensor noise and JPEG compression."""
    rng = np.random.default_rng(seed)
    x = img.astype(float) * brightness + offset
    if blur:
        x = cv2.GaussianBlur(x, (0, 0), blur)
    x = np.clip(x + rng.normal(0, noise, x.shape), 0, 255).astype(np.uint8)
    _, encoded = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return Image.fromarray(cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE))


def destroyed_qr(text: str = "NL*TNM*E12345*1") -> Image.Image:
    """A sticker whose QR code is worn away; only the printed ID is readable."""
    img = sticker(text)
    img[40:200, 80:380] = 200
    return Image.fromarray(img)


def png_bytes(image: Image.Image | np.ndarray) -> bytes:
    array = np.asarray(image)
    _, encoded = cv2.imencode(".png", array)
    return encoded.tobytes()
