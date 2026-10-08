"""
Endpoints used by the mobile app:

  POST  /v1/scans                      every scan, with its image
  PATCH /v1/scans/{scan_id}/outcome    how the driver finally found the charger
"""

import hashlib
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from pydantic import ValidationError

from app.api.images import open_image, read_upload, strip_metadata
from app.config import Settings
from app.dependencies import get_object_store, get_repository, get_settings
from app.schemas import OutcomeResponse, OutcomeUpdate, ScanCreate, ScanCreatedResponse
from app.storage.object_store import ObjectStore, build_image_key
from app.storage.repository import ScanRepository

router = APIRouter(prefix="/v1", tags=["scans"])


@router.post(
    "/scans",
    response_model=ScanCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    responses={200: {"description": "This scan_id was already stored (the app retried)"}},
)
async def create_scan(
    response: Response,
    metadata: str = Form(..., description="The scan metadata (ScanCreate) as a JSON string"),
    image: UploadFile = File(..., description="The camera frame: JPEG, PNG or WebP"),
    repo: ScanRepository = Depends(get_repository),
    store: ObjectStore = Depends(get_object_store),
    settings: Settings = Depends(get_settings),
) -> ScanCreatedResponse:
    # 1. Validate the metadata with the schema from Step 5
    try:
        scan = ScanCreate.model_validate_json(metadata)
    except ValidationError as exc:
        raise HTTPException(
            422,
            detail=exc.errors(include_url=False, include_context=False),
        )

    received_at = datetime.now(timezone.utc)

    # 2. Idempotency: if the phone retries the same scan, answer OK without storing twice
    if repo.scan_exists(scan.scan_id):
        response.status_code = status.HTTP_200_OK
        return ScanCreatedResponse(scan_id=scan.scan_id, received_at=received_at, status="duplicate")

    # 3. Check and clean the image
    raw = await read_upload(image, settings.max_image_bytes)
    clean = strip_metadata(open_image(raw))

    # 4. Store the image first, then the metadata that points to it.
    #    If the second step fails, we are left with an unused image (harmless),
    #    never with a database row pointing to a missing image.
    key = build_image_key(scan.scan_id, scan.captured_at)
    store.put_image(key, clean, content_type="image/jpeg")
    repo.save_scan(scan, image_key=key, image_sha256=hashlib.sha256(clean).hexdigest())

    return ScanCreatedResponse(scan_id=scan.scan_id, received_at=received_at)


@router.patch("/scans/{scan_id}/outcome", response_model=OutcomeResponse)
def record_outcome(
    scan_id: UUID,
    outcome: OutcomeUpdate,
    repo: ScanRepository = Depends(get_repository),
) -> OutcomeResponse:
    try:
        labelled = repo.save_outcome(scan_id, outcome)
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Scan {scan_id} not found")
    return OutcomeResponse(scan_id=scan_id, labelled=labelled)
