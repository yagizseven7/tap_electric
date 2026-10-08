"""
Watch the promotion gate (Step 12) at work, on synthetic data.

    python try_validation.py        ~10 minutes on a laptop CPU, no downloads

Story:
  0. 1,200 synthetic scans are stored and a golden test set is frozen
  1. a quickly trained, weak model     -> the gate should REJECT it
  2. a properly trained model          -> PROMOTED: it becomes the champion
  3. a retrained challenger            -> compared photo by photo with the champion

Reports are written to reports/ (one JSON + one Markdown file per candidate).

The quality bars are lower than PromotionPolicy's production defaults,
because the tiny model here learns to read from zero in a few minutes.
A fine-tuned TrOCR on real data should clear the production bars.
"""

import logging
import shutil
import time
import warnings
from pathlib import Path

from app.storage.chargers import InMemoryChargerDirectory
from app.storage.model_registry import InMemoryModelRegistry
from app.storage.object_store import InMemoryObjectStore
from app.storage.repository import InMemoryScanRepository
from app.training.dataset import CharacterCountCropper
from app.training.evaluate import format_report
from app.training.golden import build_golden_set
from app.training.job import RetrainPolicy, Validation, run_training_job
from app.training.promotion import PromotionPolicy
from app.training.synthetic import generate, store_scans
from app.training.tiny_model import build_tiny_trocr
from app.training.tracking import InMemoryTracker
from app.training.train import TrainConfig, load_trocr

# Demo bars, set for a tiny model that learns to read from zero in a few
# minutes (it reaches roughly 30% exact reads). The production defaults in
# PromotionPolicy are much stricter (80% exact, 85% resolved, 1% wrong).
DEMO_POLICY = PromotionPolicy(min_exact_match=0.20, min_resolved_rate=0.60, max_wrong_match_rate=0.03)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    for folder in ("data", "models", "reports", "golden"):
        shutil.rmtree(folder, ignore_errors=True)
    started = time.time()

    print("0. Generating 1,200 synthetic scans and freezing the golden test set...")
    # One photo per charger, 1,200 chargers: about 480 different station numbers.
    # (With fewer different IDs the model memorises them instead of learning to read.)
    chargers, scans = generate(n_chargers=1200, scans_per_charger=1, seed=7)
    repo, store = InMemoryScanRepository(), InMemoryObjectStore()
    store_scans(scans, repo, store)
    golden = build_golden_set(repo.get_labeled_scans(limit=100_000), store, "golden/v1")
    print(f"   {golden.golden_id}: {len(golden.examples)} photos, "
          f"{sum(e.hard for e in golden.examples)} of which the QR scan failed")

    registry = InMemoryModelRegistry()
    validation = Validation(golden, InMemoryChargerDirectory(chargers), DEMO_POLICY)

    def from_scratch(path: str):
        # Fresh random weights for the first model; afterwards continue from
        # the saved production model, exactly as the real job does.
        return build_tiny_trocr() if path == "tiny" else load_trocr(path)

    rounds = [
        ("1. A weak model (only 8 epochs)", TrainConfig(base_model="tiny", epochs=8, learning_rate=1e-3,
                                                        batch_size=32, augment_prob=0.3, warmup_ratio=0.05,
                                                        early_stopping_patience=8)),
        # No early stopping here: a model starting from zero first spends many
        # epochs only guessing the ID format, and would be stopped on that plateau.
        ("2. A properly trained model (50 epochs)", TrainConfig(base_model="tiny", epochs=50, learning_rate=1e-3,
                                                                batch_size=32, augment_prob=0.3, warmup_ratio=0.05,
                                                                early_stopping_patience=50)),
        ("3. Retraining from the champion (10 more epochs)", TrainConfig(base_model="tiny", epochs=10, learning_rate=3e-4,
                                                                          batch_size=32, augment_prob=0.5, warmup_ratio=0.1,
                                                                          early_stopping_patience=5)),
    ]
    for i, (title, config) in enumerate(rounds, start=1):
        if i == 3 and registry.production() is None:
            print(f"\n{title}: skipped, there is no champion to retrain from")
            continue
        print(f"\n{title}: training...")
        result = run_training_job(
            repo=repo, store=store, registry=registry, tracker=InMemoryTracker(), config=config,
            policy=RetrainPolicy(min_train_examples=50), validation=validation, data_dir="data",
            load_model=from_scratch, cropper=CharacterCountCropper(), force=True,
        )
        if result.report is None:
            print(f"   {result.status}: {result.reason}")
            continue
        print("\n" + format_report(result.report))
        print("\n" + result.decision.explain())
        production = registry.production()
        print(f"\n   -> production model is now: {production.version if production else 'none'}")

    print(f"\nDone in {(time.time() - started) / 60:.1f} min. Reports: {Path('reports').resolve()}")


if __name__ == "__main__":
    main()
