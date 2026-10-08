"""
The training job: the "system that trains and improves the model".

It runs on a schedule (e.g. every night), and each time:

    1. decide     is there enough NEW data to make retraining worthwhile?
    2. build      the dataset from all labelled scans          (Step 10)
    3. train      starting from the current production model   (Step 11)
    4. register   the result as a CANDIDATE, if it beat its starting point
                  on the validation set
    5. validate   evaluate candidate and champion on the golden test set and
                  promote the candidate only if it passes every check (Step 12)

Run it by hand:
    python -m app.training.job --force --golden-dir golden/v1

In production it would run as a scheduled job (a Kubernetes CronJob,
Airflow, or a cloud scheduler) on a machine with a GPU, separately from
the API servers, so training never slows down the scanning service.
"""

import argparse
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.inference.model import TrOCRRecognizer
from app.storage.chargers import ChargerDirectory
from app.storage.model_registry import CANDIDATE, ModelRegistry, ModelVersion
from app.storage.object_store import ObjectStore
from app.storage.repository import LabeledScan, ScanRepository
from app.training.augment import Augmenter
from app.training.dataset import Cropper, LabelGuidedCropper, OcrDataset, build_dataset
from app.training.evaluate import EvalReport, evaluate_model, format_report
from app.training.golden import GoldenSet
from app.training.promotion import PromotionDecision, PromotionPolicy, apply_decision, decide
from app.training.tracking import ExperimentTracker
from app.training.train import TrainConfig, TrainResult, load_trocr, train

logger = logging.getLogger("qr_scan_service.training")


@dataclass
class RetrainPolicy:
    """When is retraining worth it?

    Training costs GPU time, and every new model has to be validated, so we
    only retrain when enough has changed since the last model: plenty of
    new labelled scans, or a good number of new HARD ones (failed scans).
    """
    min_new_examples: int = 2000
    min_new_hard_examples: int = 300
    min_train_examples: int = 50     # below this, a dataset is too small to learn from

    def decide(self, scans: list[LabeledScan], last_model: ModelVersion | None) -> tuple[bool, str]:
        if last_model is None:
            return True, "no model has been trained yet"
        since = _aware(last_model.created_at)
        new = [s for s in scans if _aware(s.captured_at) > since]
        hard = [s for s in new if not s.decode_success]
        if len(new) >= self.min_new_examples:
            return True, f"{len(new)} new labelled scans"
        if len(hard) >= self.min_new_hard_examples:
            return True, f"{len(hard)} new failed scans"
        return False, f"only {len(new)} new scans ({len(hard)} failed) since {last_model.version}"


@dataclass
class Validation:
    """Everything step 5 needs. Leave it out to stop after registering."""
    golden: GoldenSet
    chargers: ChargerDirectory
    policy: PromotionPolicy = field(default_factory=PromotionPolicy)
    reports_dir: str = "reports"
    pipeline_settings: dict = field(default_factory=dict)  # the production pipeline thresholds


@dataclass
class JobResult:
    status: str        # skipped / not_enough_data / no_improvement / registered / promoted / rejected
    reason: str
    version: str | None = None
    dataset_id: str | None = None
    train_result: TrainResult | None = None
    dataset_counts: dict = field(default_factory=dict)
    decision: PromotionDecision | None = None
    report: EvalReport | None = None


def new_version_name() -> str:
    return f"trocr-ft-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"


def run_training_job(
    *,
    repo: ScanRepository,
    store: ObjectStore,
    registry: ModelRegistry,
    tracker: ExperimentTracker,
    config: TrainConfig,
    policy: RetrainPolicy | None = None,
    validation: Validation | None = None,
    data_dir: str = "data",
    force: bool = False,
    load_model: Callable[[str], tuple] = load_trocr,
    cropper: Cropper | None = None,
) -> JobResult:
    policy = policy or RetrainPolicy()

    # 1. Decide
    scans = repo.get_labeled_scans(limit=10_000_000)
    should_train, reason = policy.decide(scans, registry.latest())
    if not (should_train or force):
        logger.info("Skipping training: %s", reason)
        return JobResult("skipped", reason)

    # Continue from the production model if there is one: it already knows
    # our stickers. Otherwise start from the public pre-trained model.
    production = registry.production()
    init_from = production.artifact_uri if production else config.base_model
    model, processor = load_model(init_from)

    # 2. Build the dataset
    version = new_version_name()
    # By default the starting model picks the ID line on each photo. Pass
    # CharacterCountCropper() instead when the starting model can't read yet
    # (a model trained from scratch, as in the demo).
    cropper = cropper or LabelGuidedCropper(TrOCRRecognizer(model, processor, version=init_from))
    build = build_dataset(scans, store, cropper, Path(data_dir) / version)
    logger.info("Dataset %s: %s, dropped %s", build.dataset_id, build.counts(), build.dropped)

    train_rows, val_rows = build.split("train"), build.split("val")
    if len(train_rows) < policy.min_train_examples or not val_rows:
        return JobResult("not_enough_data", f"{len(train_rows)} training / {len(val_rows)} validation examples",
                         dataset_id=build.dataset_id, dataset_counts=build.counts())

    # 3. Train
    train_ds = OcrDataset(train_rows, build.directory, processor,
                          augmenter=Augmenter(p=config.augment_prob, seed=config.seed),
                          max_target_length=config.max_target_length)
    val_ds = OcrDataset(val_rows, build.directory, processor, max_target_length=config.max_target_length)
    result = train(config, model, processor, train_ds, val_ds, tracker, version,
                   extra_params={"dataset_id": build.dataset_id, "init_from": init_from, "trigger": reason})

    if not result.improved:
        return JobResult("no_improvement", "no epoch beat the starting model on validation",
                         version, build.dataset_id, result, build.counts())

    # 4. Register as a candidate
    registry.register(ModelVersion(
        version=version, artifact_uri=result.model_dir, stage=CANDIDATE,
        metrics={f"val_{k}": v for k, v in result.best_metrics.items()},
        run_id=result.run_id, dataset_id=build.dataset_id, base_model=init_from,
    ))
    if validation is None:
        return JobResult("registered", reason, version, build.dataset_id, result, build.counts())

    # 5. Validate on the golden set and promote or reject
    decision, report = validate_candidate(version, registry, validation)
    status = "promoted" if decision.promote else "rejected"
    return JobResult(status, decision.explain(), version, build.dataset_id, result, build.counts(), decision, report)


def validate_candidate(version: str, registry: ModelRegistry, validation: Validation) -> tuple[PromotionDecision, EvalReport]:
    """Evaluate the candidate AND the current champion on the same golden set,
    with the same code, then apply the promotion rules. Re-evaluating the
    champion every time (instead of reusing old numbers) guarantees a fair
    comparison even if the evaluation code or pipeline settings changed."""
    def evaluate(model: ModelVersion) -> EvalReport:
        recognizer = TrOCRRecognizer.from_pretrained(model.artifact_uri, version=model.version)
        return evaluate_model(recognizer, validation.golden, validation.chargers, model.version,
                              validation.pipeline_settings)

    challenger = evaluate(registry.get(version))
    champion_model = registry.production()
    champion = evaluate(champion_model) if champion_model else None

    decision = decide(challenger, champion, validation.policy)
    out = Path(validation.reports_dir)
    challenger.save(out)
    (out / f"{version}_decision.md").write_text(
        format_report(challenger) + ("\n\n" + format_report(champion) if champion else "")
        + "\n\n```\n" + decision.explain() + "\n```\n", encoding="utf-8")
    apply_decision(registry, decision)
    logger.info(decision.explain())
    return decision, challenger


def _aware(value: datetime) -> datetime:
    """SQLite returns times without a timezone; treat those as UTC."""
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def main() -> None:  # pragma: no cover  (needs a real database and S3)
    from app.config import Settings
    from app.main import build_object_store, build_session_factory
    from app.storage.chargers import SqlChargerDirectory
    from app.storage.model_registry import SqlModelRegistry
    from app.storage.repository import SqlScanRepository
    from app.training.golden import build_golden_set, load_golden_set
    from app.training.tracking import MlflowTracker

    parser = argparse.ArgumentParser(description="Retrain the sticker OCR model on collected scans.")
    parser.add_argument("--force", action="store_true", help="train even if the retrain policy says no")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--golden-dir", help="golden test set folder; created from the test split if it doesn't exist")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    settings = Settings.from_env()
    session_factory = build_session_factory(settings)
    if session_factory is None:
        raise SystemExit("Set DATABASE_URL: the training job reads the collected scans from the database.")
    repo, store = SqlScanRepository(session_factory), build_object_store(settings)

    validation = None
    if args.golden_dir:
        golden_dir = Path(args.golden_dir)
        golden = (load_golden_set(golden_dir) if (golden_dir / "golden.jsonl").exists()
                  else build_golden_set(repo.get_labeled_scans(limit=10_000_000), store, golden_dir))
        validation = Validation(golden, SqlChargerDirectory(session_factory))

    result = run_training_job(
        repo=repo, store=store,
        registry=SqlModelRegistry(session_factory),
        tracker=MlflowTracker(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")),
        config=TrainConfig(base_model=settings.model_name, epochs=args.epochs, output_dir=args.output_dir),
        validation=validation, data_dir=args.data_dir, force=args.force,
    )
    print(f"{result.status}: {result.reason}" + (f" -> {result.version}" if result.version else ""))


if __name__ == "__main__":
    main()
