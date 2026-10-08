"""
POST /v1/predict: run the image-to-text model on one image.

Useful for trying the model and, later, for the app to call when the
on-device QR decoder fails. (Step 9 wraps this in the full pipeline:
QR decoder -> enhancement -> model -> match against nearby chargers.)
"""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.api.images import open_image, read_upload
from app.config import Settings
from app.dependencies import get_recognizer, get_settings
from app.inference.model import TextRecognizer
from app.schemas import PredictionResponse

router = APIRouter(prefix="/v1", tags=["model"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(...),
    recognizer: TextRecognizer | None = Depends(get_recognizer),
    settings: Settings = Depends(get_settings),
) -> PredictionResponse:
    if recognizer is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded (set LOAD_MODEL=true)")

    img = open_image(await read_upload(image, settings.max_image_bytes))

    # The model is slow, ordinary (blocking) Python code. Running it in a
    # thread keeps the server free to answer other requests meanwhile.
    prediction = await run_in_threadpool(recognizer.predict, img)

    return PredictionResponse(
        text=prediction.text,
        confidence=prediction.confidence,
        model_version=prediction.model_version,
    )
