"""
POST /v1/resolve: the full pipeline. Image (+ location) in, charger out.

The app calls this when its own on-device QR scan fails for a few seconds.
"""

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.images import open_image, read_upload
from app.config import Settings
from app.dependencies import get_pipeline, get_settings
from app.inference.pipeline import PipelineResult, ScanPipeline
from app.schemas import ChargerSuggestion, ImageQualityOut, ResolveResponse

router = APIRouter(prefix="/v1", tags=["model"])


@router.post("/resolve", response_model=ResolveResponse)
async def resolve(
    image: UploadFile = File(...),
    latitude: float | None = Form(None, ge=-90, le=90),
    longitude: float | None = Form(None, ge=-180, le=180),
    gps_accuracy_m: float | None = Form(None, ge=0),
    pipeline: ScanPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
) -> ResolveResponse:
    img = open_image(await read_upload(image, settings.max_image_bytes))
    # Image processing and the model are CPU-heavy: run them off the main thread
    result = await run_in_threadpool(pipeline.resolve, img, latitude, longitude, gps_accuracy_m)
    return to_response(result)


def to_response(result: PipelineResult) -> ResolveResponse:
    q = result.quality
    return ResolveResponse(
        status=result.status.value,
        stage=result.stage,
        charger_id=result.charger.charger_id if result.charger else None,
        evse_id=result.charger.evse_id if result.charger else None,
        candidates=[
            ChargerSuggestion(charger_id=m.charger.charger_id, evse_id=m.charger.evse_id, score=round(m.score, 3))
            for m in result.candidates
        ],
        read_text=result.read_text,
        ocr_confidence=result.ocr_confidence,
        model_version=result.model_version,
        hints=result.hints,
        quality=ImageQualityOut(
            brightness=round(q.brightness, 1), contrast=round(q.contrast, 1), sharpness=round(q.sharpness, 1),
            is_dark=q.is_dark, is_overexposed=q.is_overexposed,
            is_low_contrast=q.is_low_contrast, is_blurry=q.is_blurry,
        ),
        duration_ms=round(result.duration_ms, 1),
    )
