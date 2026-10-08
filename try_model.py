"""
Try the real TrOCR model on an image of your own. Run with:

    python try_model.py path/to/photo.jpg

The first run downloads the model (~1.3 GB for trocr-base-printed) and
caches it, so later runs start much faster.

TrOCR reads ONE line of text, so crop your photo to just the printed
charger ID (or a line of text on any sticker/label) for a fair test.
"""

import sys
import time

from PIL import Image

from app.inference.model import TrOCRRecognizer

if len(sys.argv) != 2:
    sys.exit("Usage: python try_model.py path/to/photo.jpg")

print("Loading model (the first time this downloads it)...")
recognizer = TrOCRRecognizer.from_pretrained("microsoft/trocr-base-printed")
print(f"Running on: {recognizer.device}")

image = Image.open(sys.argv[1])
start = time.perf_counter()
prediction = recognizer.predict(image)
elapsed_ms = (time.perf_counter() - start) * 1000

print(f"Text:       {prediction.text!r}")
print(f"Confidence: {prediction.confidence:.2f}")
print(f"Time:       {elapsed_ms:.0f} ms")
