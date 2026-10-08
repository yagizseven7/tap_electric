"""
Generate sticker images to try POST /v1/resolve with. Run with:

    python make_test_stickers.py

Creates sample_data/stickers/ with a clean sticker and damaged versions
(dark, very dark, glare, blurry, worn-away QR code).
"""

from pathlib import Path

import cv2
import numpy as np

OUT = Path("sample_data/stickers")
TEXT = "NL*TNM*E12345*1"
rng = np.random.default_rng(42)


def sticker(text: str) -> np.ndarray:
    qr = cv2.resize(cv2.QRCodeEncoder.create().encode(text), (300, 300), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((520, 460), 255, np.uint8)
    canvas[40:340, 80:380] = qr
    cv2.putText(canvas, text, (40, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 3)
    return canvas


def photograph(img, brightness=1.0, offset=0.0, noise=0.0, blur=0.0) -> np.ndarray:
    x = img.astype(float) * brightness + offset
    if blur:
        x = cv2.GaussianBlur(x, (0, 0), blur)
    return np.clip(x + rng.normal(0, noise, x.shape), 0, 255).astype(np.uint8)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clean = sticker(TEXT)
    worn = clean.copy()
    worn[40:200, 80:380] = 200

    images = {
        "clean": clean,
        "dark": photograph(clean, brightness=0.12, noise=6),
        "very_dark": photograph(clean, brightness=0.05, noise=5),
        "glare": photograph(clean, brightness=0.12, offset=225, noise=4),
        "blurry": photograph(clean, noise=8, blur=5),
        "worn_qr": worn,
    }
    for name, img in images.items():
        path = OUT / f"{name}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        print("wrote", path)


if __name__ == "__main__":
    main()
