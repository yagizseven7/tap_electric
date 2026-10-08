"""
Shared image handling for the endpoints: size and type checks, EXIF removal.
"""

import io

from fastapi import HTTPException, UploadFile, status
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


async def read_upload(image: UploadFile, max_bytes: int) -> bytes:
    """Read the uploaded file, refusing wrong types and files that are too big."""
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Image must be one of {sorted(ALLOWED_CONTENT_TYPES)}, got {image.content_type}",
        )
    data = await image.read(max_bytes + 1)  # read one byte extra to detect "too big"
    if len(data) > max_bytes:
        raise HTTPException(413, f"Image larger than {max_bytes} bytes")
    if not data:
        raise HTTPException(422, "Image file is empty")
    return data


def open_image(data: bytes) -> Image.Image:
    """Decode bytes into an image and apply the EXIF rotation, so the image
    is upright the way the camera saw it."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()  # actually decode now, so a broken file fails here
    except (UnidentifiedImageError, OSError):
        raise HTTPException(422, "File is not a valid image")
    return ImageOps.exif_transpose(img)


def strip_metadata(img: Image.Image) -> bytes:
    """Re-encode as JPEG without EXIF metadata.

    EXIF can contain the exact GPS position, phone serial numbers and
    timestamps. We already receive the location we need in the scan
    metadata (with consent), so we drop everything hidden in the file (GDPR:
    data minimisation). Quality 95 keeps the image almost identical, which
    matters because these images become training data.
    """
    out = io.BytesIO()
    img.convert("RGB").save(out, format="JPEG", quality=95)
    return out.getvalue()
