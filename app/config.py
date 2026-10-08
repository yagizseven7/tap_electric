"""
Settings, read from environment variables.

Keeping passwords and addresses out of the code means the same code runs on
your laptop, in tests and in production; only the environment differs.

Example (Mac/Linux):
    export DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/scans
    export S3_BUCKET=qr-scans
    export LOAD_MODEL=true
"""

import os
from dataclasses import dataclass

from app.inference.model import DEFAULT_MODEL_NAME


@dataclass(frozen=True)
class Settings:
    database_url: str | None = None      # None -> in-memory repository (local development)
    s3_bucket: str | None = None         # None -> in-memory image store
    s3_endpoint_url: str | None = None   # set this to use MinIO instead of AWS S3
    model_name: str = DEFAULT_MODEL_NAME # Hugging Face name or local folder of a fine-tuned model
    load_model: bool = False             # loading the model takes time and memory; off by default
    max_image_bytes: int = 10 * 1024 * 1024  # 10 MB
    chargers_csv: str | None = None      # local testing without a database: load chargers from a CSV

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=os.getenv("DATABASE_URL"),
            s3_bucket=os.getenv("S3_BUCKET"),
            s3_endpoint_url=os.getenv("S3_ENDPOINT_URL"),
            model_name=os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME),
            load_model=os.getenv("LOAD_MODEL", "false").lower() == "true",
            max_image_bytes=int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024))),
            chargers_csv=os.getenv("CHARGERS_CSV"),
        )
