"""
Experiment tracking: a logbook of every training run.

For each run we record the settings (learning rate, dataset id, ...),
the metrics after every epoch, and the trained model files. Months later
we can still answer: "which data and settings produced the model that is
in production, and how did it score?"

Two implementations of the same contract:
  * MlflowTracker    - MLflow, the standard open-source tool. Locally it
                       writes to a SQLite file; in production it points at
                       an MLflow server that keeps files in S3.
  * InMemoryTracker  - keeps everything in lists, for tests.
"""

import math
import os
from typing import Protocol


class ExperimentTracker(Protocol):
    def start(self, run_name: str, params: dict) -> str: ...

    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...

    def log_artifacts(self, directory: str) -> None: ...

    def end(self, status: str = "FINISHED") -> None: ...


class MlflowTracker:
    """Example tracking URIs:
        sqlite:///mlflow.db            a local file (view with: mlflow ui --backend-store-uri sqlite:///mlflow.db)
        http://mlflow.internal:5000    a shared MLflow server
    """

    def __init__(self, tracking_uri: str | None = None, experiment: str = "qr-sticker-ocr"):
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import mlflow  # imported here so the API service doesn't need MLflow installed

        self.mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

    def start(self, run_name: str, params: dict) -> str:
        run = self.mlflow.start_run(run_name=run_name)
        # MLflow limits parameter values to 6000 characters
        self.mlflow.log_params({k: str(v)[:6000] for k, v in params.items()})
        return run.info.run_id

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.mlflow.log_metrics({k: float(v) for k, v in metrics.items() if math.isfinite(v)}, step=step)

    def log_artifacts(self, directory: str) -> None:
        self.mlflow.log_artifacts(directory, artifact_path="model")

    def end(self, status: str = "FINISHED") -> None:
        self.mlflow.end_run(status=status)


class InMemoryTracker:
    def __init__(self) -> None:
        self.runs: list[dict] = []

    def start(self, run_name: str, params: dict) -> str:
        self.runs.append({"name": run_name, "params": dict(params), "metrics": [], "artifacts": [], "status": None})
        return f"run-{len(self.runs)}"

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.runs[-1]["metrics"].append({"step": step, **metrics})

    def log_artifacts(self, directory: str) -> None:
        self.runs[-1]["artifacts"].append(directory)

    def end(self, status: str = "FINISHED") -> None:
        self.runs[-1]["status"] = status
