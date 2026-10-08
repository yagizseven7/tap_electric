"""
Classical image processing (no machine learning): make a bad photo easier
to read, and find the regions that probably contain printed text.

Classical methods are fast, free and predictable, so the pipeline tries
them before calling the (slower) model.
"""

from collections.abc import Iterator

import cv2
import numpy as np
from PIL import Image

MAX_SIDE = 1280  # bigger images are slower to process without being easier to read


def to_gray(image: Image.Image) -> np.ndarray:
    """PIL image (any mode) -> greyscale numpy array, at most MAX_SIDE pixels."""
    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    scale = MAX_SIDE / max(height, width)
    if scale < 1:
        gray = cv2.resize(gray, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return gray


def clahe(gray: np.ndarray) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization.

    Boosts contrast separately in small tiles of the image, so a sticker
    that is dark on one side and lit on the other is fixed on both sides.
    Good against darkness and fading.
    """
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)


def gamma(gray: np.ndarray, value: float) -> np.ndarray:
    """value < 1 brightens dark areas; value > 1 darkens bright areas."""
    table = ((np.arange(256) / 255.0) ** value * 255).astype(np.uint8)
    return cv2.LUT(gray, table)


def sharpen(gray: np.ndarray) -> np.ndarray:
    """Unsharp mask: subtract a blurred copy to make edges stand out."""
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
    return cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Turn every pixel black or white, with a threshold that adapts to the
    local brightness. QR decoders work on black/white anyway, and this
    removes uneven lighting and shadows."""
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)


def stretch(gray: np.ndarray) -> np.ndarray:
    """Spread the pixel values over the full 0-255 range, ignoring the
    darkest and brightest 1% (so a single reflection doesn't spoil it)."""
    low, high = np.percentile(gray, (1, 99))
    stretched = (gray.astype(np.float32) - low) * 255.0 / max(high - low, 1.0)
    return np.clip(stretched, 0, 255).astype(np.uint8)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Remove sensor noise. In dark photos the camera amplifies the signal,
    and the noise along with it; that grain breaks the QR pattern.
    Non-local means averages similar-looking patches, so it removes noise
    while keeping the sharp QR edges. Slower than the other variants."""
    return stretch(cv2.fastNlMeansDenoising(gray, None, h=7))


def enhanced_variants(gray: np.ndarray) -> Iterator[tuple[str, np.ndarray]]:
    """Variants of the image to retry QR decoding on, cheapest and most
    often helpful first. A generator: we stop as soon as one works."""
    equalized = clahe(gray)
    yield "clahe", equalized
    yield "brighten", gamma(gray, 0.5)
    yield "darken", gamma(gray, 2.0)
    yield "smooth", clahe(cv2.GaussianBlur(gray, (0, 0), 1.5))  # light noise removal
    yield "sharpen", sharpen(equalized)
    yield "binarize", binarize(equalized)
    yield "denoise", denoise(gray)                               # heavy noise removal, slowest


def find_text_lines(gray: np.ndarray, max_lines: int = 5, padding: float = 0.15) -> list[np.ndarray]:
    """Find horizontal strips that probably contain one line of printed text,
    and return them as crops of the ORIGINAL (not enhanced) image, biggest first.

    TrOCR reads one line at a time, so we must crop first. Idea: characters
    have many small edges close together. We find edges, smear them
    horizontally so the letters of a word merge into one blob, and keep
    blobs that are wide and flat like a line of text. (The QR code itself
    is square, so it is filtered out.)

    Detection runs on a contrast-enhanced copy (so it also works on dark
    photos), but the crops come from the original image. The training code
    (Step 10) can then damage a crop first and prepare it afterwards, in
    the same order as in real life.

    This is deliberately simple. A production version would use a trained
    text detector (e.g. Florence-2 or a PaddleOCR detector), which is an
    obvious next improvement.
    """
    return [
        gray[max(0, y): y + h, max(0, x): x + w]
        for x, y, w, h in text_line_boxes(gray, max_lines, padding)
    ]


def prepare_for_model(crop: np.ndarray) -> Image.Image:
    """The exact preparation a text crop gets before the model reads it.

    Used in BOTH the live pipeline and training. If the two prepared images
    differently, the model would be trained on pictures that look different
    from the ones it sees in use ("training/serving skew"), and its measured
    accuracy would not hold in production.

    A global contrast stretch fixes dark and faded crops without the tile
    artefacts that CLAHE gives on very small images.
    """
    return Image.fromarray(stretch(crop)).convert("RGB")


def text_line_boxes(gray: np.ndarray, max_lines: int = 5, padding: float = 0.15) -> list[tuple[int, int, int, int]]:
    """Bounding boxes (x, y, width, height) of likely text lines, with some
    padding so the first and last characters are not cut off."""
    height, width = gray.shape
    # Even out the brightness, then blur away sensor noise. Without the blur,
    # the grain of a dark photo produces thousands of tiny "edges" and the
    # text drowns in them. (Local contrast boosting such as CLAHE makes this
    # worse: it amplifies the grain in flat areas.)
    detect_on = cv2.GaussianBlur(stretch(gray), (0, 0), max(1.0, min(height, width) / 300))
    edges = cv2.morphologyEx(detect_on, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, mask = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, width // 40), 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pieces = [cv2.boundingRect(c) for c in contours]
    pieces = [b for b in pieces if b[3] >= 0.015 * height]   # ignore specks of noise

    boxes = []
    for x, y, w, h in merge_same_row(pieces):
        looks_like_text_line = (
            w >= 2.5 * h                     # wide and flat
            and h <= 0.2 * height
            and w >= 0.1 * width
        )
        if looks_like_text_line:
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)  # biggest first
    padded = []
    for x, y, w, h in boxes[:max_lines]:
        pad_x, pad_y = round(w * padding / 2), round(h * padding * 2)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(width, x + w + pad_x), min(height, y + h + pad_y)
        padded.append((x0, y0, x1 - x0, y1 - y0))
    return padded


Box = tuple[int, int, int, int]


def merge_same_row(boxes: list[Box], max_gap: float = 1.5) -> list[Box]:
    """Join boxes that sit on the same row and are close together.

    The horizontal smear joins most letters of a word, but a wide gap (for
    example after "NL" in "NL*LMS*E47268*4") can leave a piece on its own.
    Without this step that piece is too small to look like a text line, is
    thrown away, and the ID loses its first characters.
    """
    boxes = list(boxes)
    merged = True
    while merged:  # keep merging until nothing changes
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _same_row_and_close(boxes[i], boxes[j], max_gap):
                    boxes[i] = _union(boxes[i], boxes[j])
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes


def _same_row_and_close(a: Box, b: Box, max_gap: float) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    vertical_overlap = min(ay + ah, by + bh) - max(ay, by)
    gap = max(bx - (ax + aw), ax - (bx + bw), 0)
    similar_height = min(ah, bh) >= 0.4 * max(ah, bh)
    return similar_height and vertical_overlap >= 0.5 * min(ah, bh) and gap <= max_gap * max(ah, bh)


def _union(a: Box, b: Box) -> Box:
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return (x0, y0, x1 - x0, y1 - y0)


def character_counts(crop: np.ndarray) -> list[int]:
    """Count shapes in a text-line crop that look like letters or digits,
    at several darkness levels.

    Pixels darker than a threshold count as ink, then every connected blob
    is checked: a character is roughly as tall as the line and not much
    wider than it is tall. Small marks (asterisks, specks of noise) and
    long bars (pieces of a QR code) don't count.

    Why several thresholds? In a blurry photo neighbouring letters melt
    together at the normal threshold and the count drops to 0 or 1. Only
    the darkest cores of the letters are still separate, so a stricter
    threshold counts them correctly.
    """
    if crop.size == 0:
        return [0]
    stretched = stretch(crop)
    otsu, _ = cv2.threshold(stretched, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    line_height = crop.shape[0]
    counts = []
    for threshold in (otsu, otsu * 0.75, otsu * 0.5, otsu * 0.3):
        ink = (stretched < threshold).astype(np.uint8) * 255
        _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        counts.append(sum(
            1 for x, y, w, h, area in stats[1:]  # row 0 is the background
            if 0.25 * line_height <= h <= 0.95 * line_height and w <= 1.5 * h and area >= 0.15 * w * h
        ))
    return counts
