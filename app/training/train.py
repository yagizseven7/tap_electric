"""
Step 11: fine-tuning the model on collected scans.

Fine-tuning = starting from a model that can already read printed text
(TrOCR) and training it a little further on OUR data: charger IDs, our
sticker fonts, our lighting problems. Much less data and time is needed
than training from zero.

One training run:

    measure the starting model on the validation set   ("baseline")
    for each epoch (= one pass over the training data):
        for each batch of examples:
            1. forward:   the model predicts the label, token by token
            2. loss:      how wrong it was (cross-entropy)
            3. backward:  compute in which direction each weight should move
            4. step:      move every weight a tiny bit in that direction
        measure on the validation set
        better than ever before? -> save this version of the model
        no improvement for `patience` epochs? -> stop early
    result: the best version, its metrics, and whether it beat the baseline

The validation set is never trained on. It shows whether the model
learned to READ (it improves on unseen stickers) or only MEMORISED the
training images (it improves on training data but not on validation:
"overfitting"). Early stopping stops right before that happens.

Written as a plain PyTorch loop rather than Hugging Face's Trainer class,
so every step is visible and explainable; the Trainer does the same
things behind the scenes.
"""

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_linear_schedule_with_warmup

from app.training.dataset import OcrDataset, collate_batch, sample_weights
from app.training.metrics import ocr_metrics
from app.training.tracking import ExperimentTracker


@dataclass
class TrainConfig:
    base_model: str = "microsoft/trocr-base-printed"
    output_dir: str = "models"
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 5e-5      # small: we only want to adjust a model that already reads well
    weight_decay: float = 0.01       # gently pulls weights toward zero, against overfitting
    warmup_ratio: float = 0.1        # first 10% of steps: ramp the learning rate up from 0
    max_grad_norm: float = 1.0       # cap on the size of one update, against unstable jumps
    max_target_length: int = 32      # tokens; charger IDs are short
    augment_prob: float = 0.8        # share of training crops that get random damage
    hard_example_weight: float = 3.0 # failed scans are drawn 3x as often
    freeze_encoder: bool = False     # True: only train the text decoder (faster, safer with little data)
    early_stopping_patience: int = 3
    min_improvement: float = 0.001   # CER must drop by at least this much to count as better
    eval_batch_size: int = 32
    seed: int = 42
    device: str | None = None        # None = GPU if available, else CPU


@dataclass
class TrainResult:
    version: str
    improved: bool               # did any epoch beat the starting model on validation?
    model_dir: str | None        # where the best model was saved (None if not improved)
    baseline_metrics: dict
    best_metrics: dict
    best_epoch: int
    run_id: str
    history: list[dict] = field(default_factory=list)
    seconds: float = 0.0


def load_trocr(name_or_path: str) -> tuple[VisionEncoderDecoderModel, TrOCRProcessor]:
    """A Hugging Face name ('microsoft/trocr-base-printed') or a local folder
    of a model we trained before."""
    return VisionEncoderDecoderModel.from_pretrained(name_or_path), TrOCRProcessor.from_pretrained(name_or_path)


def set_seed(seed: int) -> None:
    """Same seed -> same random choices -> a run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fill_special_tokens(model: VisionEncoderDecoderModel, tokenizer) -> None:
    """The model must know which token starts a text, ends it, and is padding.
    Pre-trained TrOCR models already have these; we only fill in what's
    missing and never overwrite, so we keep the convention the model was
    originally trained with."""
    defaults = {
        "decoder_start_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    for config in (model.config, model.generation_config):
        for name, value in defaults.items():
            if getattr(config, name, None) is None:
                setattr(config, name, value)


@torch.inference_mode()
def evaluate(model, tokenizer, loader: DataLoader, device: str, max_new_tokens: int = 32) -> tuple[dict, list[str]]:
    """Read every validation crop and compare with the labels.

    Uses greedy decoding (num_beams=1): faster than the beam search used in
    production, which is fine here because we only compare model versions
    with each other. Step 12 measures the real production setting.
    """
    model.eval()
    predictions, targets = [], []
    for batch in loader:
        generated = model.generate(batch["pixel_values"].to(device), max_new_tokens=max_new_tokens, num_beams=1)
        predictions += [t.strip() for t in tokenizer.batch_decode(generated, skip_special_tokens=True)]
        targets += batch["texts"]
    return ocr_metrics(predictions, targets), predictions


def train(
    config: TrainConfig,
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    train_ds: OcrDataset,
    val_ds: OcrDataset,
    tracker: ExperimentTracker,
    version: str,
    extra_params: dict | None = None,
) -> TrainResult:
    started = time.time()
    set_seed(config.seed)
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = processor.tokenizer
    fill_special_tokens(model, tokenizer)
    model.to(device)

    if config.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    # Oversample hard examples: they are drawn more often within each epoch
    generator = torch.Generator().manual_seed(config.seed)
    sampler = WeightedRandomSampler(sample_weights(train_ds.rows, config.hard_example_weight),
                                    num_samples=len(train_ds), replacement=True, generator=generator)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=config.eval_batch_size, shuffle=False, collate_fn=collate_batch)

    # AdamW: the standard optimizer for Transformers. It adapts the step size
    # per weight, and weight_decay keeps weights small (against overfitting).
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = max(1, config.epochs * len(train_loader))
    # Learning rate schedule: ramp up (warmup), then decrease linearly to 0.
    # Big sudden updates at the start could damage what the model already knows.
    scheduler = get_linear_schedule_with_warmup(optimizer, int(config.warmup_ratio * total_steps), total_steps)

    # Mixed precision on a GPU: some maths in 16-bit -> about 2x faster, less memory
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_id = tracker.start(version, {
        **asdict(config), **(extra_params or {}),
        "train_examples": len(train_ds), "val_examples": len(val_ds), "device": device,
    })

    try:
        baseline, _ = evaluate(model, tokenizer, val_loader, device, config.max_target_length)
        tracker.log_metrics({f"val_{k}": v for k, v in baseline.items()}, step=0)
        best, best_epoch, epochs_without_improvement = baseline, 0, 0
        history = [{"epoch": 0, **{f"val_{k}": v for k, v in baseline.items()}}]
        model_dir = Path(config.output_dir) / version

        for epoch in range(1, config.epochs + 1):
            model.train()  # switch on training behaviour (e.g. dropout)
            losses = []
            for batch in train_loader:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    # Given the labels, the model computes the loss itself:
                    # how surprised it was by each correct next character.
                    loss = model(pixel_values=batch["pixel_values"].to(device),
                                 labels=batch["labels"].to(device)).loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(loss.item())

            val, _ = evaluate(model, tokenizer, val_loader, device, config.max_target_length)
            row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "learning_rate": scheduler.get_last_lr()[0],
                   **{f"val_{k}": v for k, v in val.items()}}
            history.append(row)
            tracker.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)

            if val["cer"] < best["cer"] - config.min_improvement:
                best, best_epoch, epochs_without_improvement = val, epoch, 0
                model.save_pretrained(model_dir)      # keep only the best version
                processor.save_pretrained(model_dir)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.early_stopping_patience:
                    break

        improved = best_epoch > 0
        if improved:
            (model_dir / "training_metrics.json").write_text(json.dumps(
                {"baseline": baseline, "best": best, "best_epoch": best_epoch, "history": history}, indent=2))
            tracker.log_artifacts(str(model_dir))
        tracker.end("FINISHED")
    except Exception:
        tracker.end("FAILED")
        raise

    return TrainResult(
        version=version, improved=improved, model_dir=str(model_dir) if improved else None,
        baseline_metrics=baseline, best_metrics=best, best_epoch=best_epoch, run_id=run_id,
        history=history, seconds=time.time() - started,
    )
