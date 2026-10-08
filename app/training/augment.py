"""
Data augmentation: turn good photos into realistic bad ones.

Why: most scans succeed, so most labelled data consists of EASY images.
But the model is only used when the QR code fails, so it must be good at
HARD images: dark, blurry, glare, faded, dirty. We create hard versions of
the easy images during training. Every epoch the model sees a different
random damage of the same crop, so it learns to read through the damage
instead of memorising the picture.

The damage is applied in the order it happens physically:

    scene       the sticker itself      faded, dirty, in a shadow
    optics      the lens                tilted, out of focus, motion blur
    exposure    the light               too dark or too bright
    sensor      the camera chip         noise, low resolution
    processing  the phone software      JPEG compression

Applying them in this order gives more realistic images than a random
order (e.g. noise is added AFTER darkening, as in a real camera).

Every function takes a greyscale text-line crop (uint8, typically 30-60
pixels high, the model's input) and returns an image of the same shape.
The ranges are starting values: tune them by comparing the brightness,
contrast and sharpness of augmented crops with those of real FAILED scans
(quality.py measures all three).
"""

from collections.abc import Callable

import cv2
import numpy as np

Degradation = Callable[[np.ndarray, np.random.Generator], np.ndarray]

REFERENCE_HEIGHT = 40  # pixel sizes below are for a 40 px high crop and scale with the crop


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def _scale(img: np.ndarray) -> float:
    return float(np.clip(img.shape[0] / REFERENCE_HEIGHT, 0.5, 3.0))


# --- scene ------------------------------------------------------------------

def fade(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sun-bleached sticker: the print blends into the paper."""
    alpha = rng.uniform(0.3, 0.75)
    paper = rng.uniform(170, 235)
    return _clip(img * (1 - alpha) + paper * alpha)


def dirt(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Scratches, dirt and rain drops on the sticker."""
    out = img.copy()
    height, width = img.shape
    s = _scale(img)
    for _ in range(int(rng.integers(1, 5))):
        colour = int(rng.uniform(40, 200))
        if rng.random() < 0.5:   # scratch
            p1 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            p2 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            cv2.line(out, p1, p2, colour, max(1, round(rng.uniform(1, 2) * s)))
        else:                    # blob
            centre = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            axes = (max(1, round(rng.uniform(2, 6) * s)), max(1, round(rng.uniform(1, 4) * s)))
            cv2.ellipse(out, centre, axes, float(rng.uniform(0, 180)), 0, 360, colour, -1)
    return out


def shadow(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The shadow of the driver or the phone over part of the sticker."""
    height, width = img.shape
    darkest = rng.uniform(0.25, 0.6)
    ramp = np.linspace(darkest, 1.0, width)
    if rng.random() < 0.5:
        ramp = ramp[::-1]
    return _clip(img * ramp[None, :])


# --- optics -----------------------------------------------------------------

def tilt(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phone not held straight in front of the sticker: perspective + rotation."""
    height, width = img.shape
    jitter = np.array([width * 0.04, height * 0.12])
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    moved = (corners + rng.uniform(-1, 1, corners.shape) * jitter).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(corners, moved)
    return cv2.warpPerspective(img, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


def gaussian_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Out of focus (the camera focused on the background)."""
    return cv2.GaussianBlur(img, (0, 0), rng.uniform(0.6, 2.0) * _scale(img))


def motion_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The hand moved while the photo was taken."""
    length = max(3, round(rng.uniform(3, 9) * _scale(img)))
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (length, length))
    return cv2.filter2D(img, -1, kernel / max(kernel.sum(), 1e-6), borderType=cv2.BORDER_REPLICATE)


# --- exposure ---------------------------------------------------------------

def darken(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Low light. The image gets darker AND noisier: the camera amplifies
    the weak signal, and the sensor noise along with it."""
    dark = img * rng.uniform(0.08, 0.4)
    return _clip(dark + rng.normal(0, rng.uniform(2, 8), img.shape))


def overexpose(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Glare or flash reflection: everything pushed toward white."""
    return _clip(img * rng.uniform(0.15, 0.5) + rng.uniform(140, 225))


# --- sensor -----------------------------------------------------------------

def noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Grainy sensor noise."""
    return _clip(img + rng.normal(0, rng.uniform(4, 15), img.shape))


def low_resolution(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Photo taken from too far away: few pixels per character."""
    height, width = img.shape
    factor = rng.uniform(0.3, 0.7)
    small = cv2.resize(img, (max(1, round(width * factor)), max(1, round(height * factor))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


# --- processing -------------------------------------------------------------

def jpeg(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Heavy JPEG compression (blocky artefacts)."""
    _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(15, 60))])
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


STAGES: list[tuple[str, list[Degradation]]] = [
    ("scene", [fade, dirt, shadow]),
    ("optics", [tilt, gaussian_blur, motion_blur]),
    ("exposure", [darken, overexpose]),
    ("sensor", [noise, low_resolution]),
    ("processing", [jpeg]),
]


class Augmenter:
    """Randomly damages a crop.

    p           chance that a crop is damaged at all (the rest stay clean, so
                the model keeps reading clean stickers well too)
    stage_p     chance that each stage applies one of its damages
    """

    def __init__(self, p: float = 0.8, stage_p: float = 0.4, seed: int | None = None):
        self.p = p
        self.stage_p = stage_p
        self.rng = np.random.default_rng(seed)

    def __call__(self, gray: np.ndarray) -> np.ndarray:
        if self.rng.random() >= self.p:
            return gray

        chosen = [stage for stage in STAGES if self.rng.random() < self.stage_p]
        if not chosen:  # make sure a damaged crop is actually damaged
            chosen = [STAGES[int(self.rng.integers(len(STAGES)))]]

        out = gray
        for _, options in chosen:  # STAGES order = physical order
            out = options[int(self.rng.integers(len(options)))](out, self.rng)
        return out
