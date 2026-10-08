"""
Image storage.

Images are large binary files, so they go to object storage (Amazon S3, or
MinIO which offers the same interface and can run locally). The database
only stores the key, e.g. "scans/2026/10/06/<scan_id>.jpg".

Two implementations share the same methods:
  * S3ObjectStore       - the real one (needs an S3 bucket)
  * InMemoryObjectStore - keeps images in a dictionary, for tests
"""

from datetime import datetime
from typing import Protocol
from uuid import UUID


def build_image_key(scan_id: UUID, captured_at: datetime, extension: str = "jpg") -> str:
    """Date folders make it easy to delete old images (retention policy)
    and to list one day's data."""
    return f"scans/{captured_at:%Y/%m/%d}/{scan_id}.{extension}"


class ObjectStore(Protocol):
    """The 'contract': any image store must offer these two methods."""

    def put_image(self, key: str, data: bytes, content_type: str = "image/jpeg") -> None: ...

    def get_image(self, key: str) -> bytes: ...


class S3ObjectStore:
    """Stores images in an S3 bucket (or MinIO when endpoint_url is given)."""

    def __init__(self, bucket: str, endpoint_url: str | None = None):
        import boto3  # imported here so tests don't need boto3 installed

        self.bucket = bucket
        self.client = boto3.client("s3", endpoint_url=endpoint_url)

    def put_image(self, key: str, data: bytes, content_type: str = "image/jpeg") -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            ServerSideEncryption="AES256",  # images contain location context: encrypt at rest
        )

    def get_image(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()


class InMemoryObjectStore:
    """Fake store for tests and local development."""

    def __init__(self) -> None:
        self._images: dict[str, bytes] = {}

    def put_image(self, key: str, data: bytes, content_type: str = "image/jpeg") -> None:
        self._images[key] = data

    def get_image(self, key: str) -> bytes:
        if key not in self._images:
            raise KeyError(f"No image stored under {key}")
        return self._images[key]
