"""
Dependencies: small functions FastAPI calls to hand each endpoint what it needs.

The endpoints say `repo = Depends(get_repository)` instead of creating a
database connection themselves. In tests we put fake versions on app.state,
and the endpoints don't notice the difference.
"""

from fastapi import Request

from app.config import Settings
from app.inference.model import TextRecognizer
from app.inference.pipeline import ScanPipeline
from app.storage.object_store import ObjectStore
from app.storage.repository import ScanRepository


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_repository(request: Request) -> ScanRepository:
    return request.app.state.repository


def get_object_store(request: Request) -> ObjectStore:
    return request.app.state.object_store


def get_recognizer(request: Request) -> TextRecognizer | None:
    return request.app.state.recognizer


def get_pipeline(request: Request) -> ScanPipeline:
    return request.app.state.pipeline
