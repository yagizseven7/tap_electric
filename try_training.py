"""
Run the whole training system on your laptop with synthetic data.

    python try_training.py           tiny model from scratch: no download, ~5-10 minutes on a CPU
    python try_training.py --real    fine-tune microsoft/trocr-small-printed
                                     (downloads ~250 MB; slow on a CPU, a GPU helps a lot)

What happens:
  1. 1,200 synthetic scans (400 chargers) are generated and stored through
     the same repository code the API uses
  2. the training job builds the dataset (Step 10) and trains (Step 11)
  3. the result is registered as a candidate model
  4. the new model reads a few validation stickers it never trained on

Every run is logged to MLflow. Afterwards, look at the charts with:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
and open http://localhost:5000
"""

import argparse
import logging
import time
import warnings

import numpy as np
from PIL import Image

from app.inference.enhance import prepare_for_model
from app.inference.model import TrOCRRecognizer
from app.storage.model_registry import InMemoryModelRegistry
from app.storage.object_store import InMemoryObjectStore
from app.storage.repository import InMemoryScanRepository
from app.training.dataset import CharacterCountCropper, load_manifest
from app.training.job import run_training_job
from app.training.synthetic import generate, store_scans
from app.training.tiny_model import build_tiny_trocr
from app.training.tracking import InMemoryTracker
from app.training.train import TrainConfig, load_trocr


class PrintingTracker:
    """Passes everything on to another tracker and prints each epoch."""

    def __init__(self, inner):
        self.inner = inner
        self.started = time.time()

    def start(self, run_name, params):
        print(f"\n{'epoch':>5} {'train loss':>10} {'val CER':>8} {'val exact':>9} {'time':>6}")
        return self.inner.start(run_name, params)

    def log_metrics(self, metrics, step):
        loss = f"{metrics['train_loss']:.3f}" if "train_loss" in metrics else "-"
        print(f"{step:>5} {loss:>10} {metrics['val_cer']:>8.1%} {metrics['val_exact_match']:>9.1%} "
              f"{time.time() - self.started:>5.0f}s", flush=True)
        self.inner.log_metrics(metrics, step)

    def log_artifacts(self, directory):
        self.inner.log_artifacts(directory)

    def end(self, status="FINISHED"):
        self.inner.end(status)


def make_tracker():
    try:
        from app.training.tracking import MlflowTracker
        return MlflowTracker("sqlite:///mlflow.db"), True
    except ImportError:
        print("MLflow is not installed; metrics are only printed (pip install mlflow).")
        return InMemoryTracker(), False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="fine-tune the real pre-trained TrOCR")
    parser.add_argument("--chargers", type=int, default=1200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")

    print(f"1. Generating {args.chargers} synthetic scans...")
    # One photo per charger keeps many DIFFERENT IDs (about 480 stations). With
    # few different IDs the model memorises them instead of learning to read.
    _, scans = generate(n_chargers=args.chargers, scans_per_charger=1, seed=7)
    repo, store = InMemoryScanRepository(), InMemoryObjectStore()
    store_scans(scans, repo, store)
    failed = sum(not s.scan.decode.success for s in scans)
    print(f"   {len(scans)} scans stored, of which {failed} failed to decode on the 'phone'")

    if args.real:
        # A pre-trained model that already reads: a few epochs, small learning rate.
        # The default cropper lets this model find the ID line on each photo.
        config = TrainConfig(base_model="microsoft/trocr-small-printed", epochs=5, batch_size=8,
                             learning_rate=5e-5, early_stopping_patience=2)
        load_model, cropper = load_trocr, None
    else:
        # A model with random weights: it must learn to read from zero, which
        # needs a much higher learning rate, many epochs, and lighter damage.
        # It first spends many epochs only guessing the ID format before it
        # starts reading, so early stopping is switched off (patience = epochs).
        config = TrainConfig(base_model="tiny-from-scratch", epochs=40, batch_size=32, learning_rate=1e-3,
                             warmup_ratio=0.05, augment_prob=0.3, early_stopping_patience=40)
        load_model, cropper = (lambda name: build_tiny_trocr()), CharacterCountCropper()

    tracker, using_mlflow = make_tracker()
    print("2. Building the dataset and training (Step 10 + 11)...")
    result = run_training_job(
        repo=repo, store=store, registry=InMemoryModelRegistry(), tracker=PrintingTracker(tracker),
        config=config, data_dir="data", load_model=load_model, cropper=cropper, force=True,
    )

    print(f"\n3. Result: {result.status} ({result.reason})")
    print(f"   dataset {result.dataset_id}: {result.dataset_counts}")
    if result.train_result is None:
        return
    tr = result.train_result
    print(f"   validation CER: {tr.baseline_metrics['cer']:.1%} before -> {tr.best_metrics['cer']:.1%} after "
          f"(best epoch {tr.best_epoch}, {tr.seconds / 60:.1f} min)")
    if not tr.improved:
        return
    print(f"   model saved in {tr.model_dir}")

    print("\n4. The new model reads validation stickers it never trained on:")
    recognizer = TrOCRRecognizer.from_pretrained(tr.model_dir, version=result.version, num_beams=4)
    dataset_dir = f"data/{result.version}"
    for row in [r for r in load_manifest(dataset_dir) if r.split == "val"][:8]:
        crop = np.asarray(Image.open(f"{dataset_dir}/{row.crop_file}").convert("L"))
        p = recognizer.predict(prepare_for_model(crop))
        mark = "OK " if p.text.replace("*", "") == row.text.replace("*", "") else "   "
        print(f"   {mark} truth {row.text:<18} read {p.text:<18} confidence {p.confidence:.2f}")

    if using_mlflow:
        print("\nCharts: mlflow ui --backend-store-uri sqlite:///mlflow.db  ->  http://localhost:5000")


if __name__ == "__main__":
    main()
