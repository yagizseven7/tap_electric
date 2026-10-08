"""
Image quality measurements.

Three uses:
  1. Choose which enhancements to try (dark image -> brighten it).
  2. Tell the driver what to fix ("too dark, turn on the flashlight").
     Today the app can't help drivers at all when a scan fails; this is
     the cheapest way to help them right away.
  3. Group results by condition when validating the model (Step 12):
     "how accurate are we on dark images?"

All measurements are taken on a greyscale image resized to a fixed width,
so the numbers are comparable between a 12-megapixel and a 2-megapixel phone.
"""

from dataclasses import dataclass

import cv2
import numpy as np

REFERENCE_WIDTH = 640

# Starting thresholds. They are guesses: tune them on collected data by
# checking which values separate successful from failed scans.
DARK_BELOW = 60          # mean brightness, 0-255
BRIGHT_ABOVE = 210
OVEREXPOSED_FRACTION = 0.25
LOW_CONTRAST_BELOW = 25  # standard deviation of brightness
BLURRY_BELOW = 15        # variance of the Laplacian, after contrast normalisation


@dataclass(frozen=True)
class ImageQuality:
    brightness: float           # average pixel value: 0 = black, 255 = white
    contrast: float             # spread of pixel values: low = washed out / faded
    sharpness: float            # high = crisp edges, low = blurry
    overexposed_fraction: float # share of pixels that are (almost) pure white

    @property
    def is_dark(self) -> bool:
        return self.brightness < DARK_BELOW

    @property
    def is_overexposed(self) -> bool:
        # A white sticker is naturally bright; it's only a problem when the
        # brightness also washes out the contrast.
        too_bright = self.brightness > BRIGHT_ABOVE or self.overexposed_fraction > OVEREXPOSED_FRACTION
        return too_bright and self.is_low_contrast

    @property
    def is_low_contrast(self) -> bool:
        return self.contrast < LOW_CONTRAST_BELOW

    @property
    def is_blurry(self) -> bool:
        return self.sharpness < BLURRY_BELOW

    def hints(self) -> list[str]:
        """Short instructions the app can show to the driver."""
        hints = []
        if self.is_dark:
            hints.append("It's too dark. Turn on the flashlight.")
        if self.is_overexposed:
            hints.append("There's glare on the sticker. Tilt your phone slightly.")
        if self.is_blurry:
            hints.append("The image is blurry. Hold your phone still, about 15 cm from the sticker.")
        if self.is_low_contrast and not (self.is_dark or self.is_overexposed):
            hints.append("The sticker looks faded. Try typing the ID printed on it.")
        return hints


def to_reference_size(gray: np.ndarray) -> np.ndarray:
    height, width = gray.shape[:2]
    if width == REFERENCE_WIDTH:
        return gray
    new_height = max(1, round(height * REFERENCE_WIDTH / width))
    return cv2.resize(gray, (REFERENCE_WIDTH, new_height), interpolation=cv2.INTER_AREA)


def measure_sharpness(small: np.ndarray) -> float:
    """Variance of the Laplacian, a standard blur measure.

    The Laplacian highlights edges. A sharp image has strong edges in some
    places and none in others -> high variance. Blur smooths all edges away
    -> low variance.

    Two corrections make it fair across lighting conditions:
      * contrast normalisation: a dark photo has weak edges even when it is
        in focus, so we first rescale every image to the same contrast;
      * a light blur: sensor noise looks like thousands of tiny "edges"
        and would make a blurry, noisy image look sharp.
    """
    pixels = small.astype(np.float64)
    normalized = (pixels - pixels.mean()) / max(pixels.std(), 1.0) * 50 + 128
    smoothed = cv2.GaussianBlur(normalized, (0, 0), 1.0)
    return float(cv2.Laplacian(smoothed, cv2.CV_64F).var())


def measure_quality(gray: np.ndarray) -> ImageQuality:
    small = to_reference_size(gray)
    return ImageQuality(
        brightness=float(small.mean()),
        contrast=float(small.std()),
        sharpness=measure_sharpness(small),
        overexposed_fraction=float((small >= 250).mean()),
    )
