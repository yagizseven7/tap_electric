"""
The image-to-text model: reads the charger ID printed on a QR sticker.

We use TrOCR (Microsoft), an open-source model from Hugging Face:
  * microsoft/trocr-base-printed   (~334M parameters, more accurate)  <- default
  * microsoft/trocr-small-printed  (~62M parameters, faster on CPU)

TrOCR has two halves:
  * an ENCODER (a Vision Transformer) that turns the image into numbers
  * a DECODER (a text Transformer) that turns those numbers into text,
    one token (piece of text) at a time

Everything model-specific is hidden behind the TextRecognizer contract, so
the API and pipeline only ever call .predict(image). Swapping TrOCR for
another model (e.g. Florence-2) later means writing one new class.
"""

from dataclasses import dataclass
from typing import Protocol

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

DEFAULT_MODEL_NAME = "microsoft/trocr-base-printed"


@dataclass(frozen=True)
class Prediction:
    text: str
    confidence: float   # 0.0 - 1.0; how sure the model is about the whole text
    model_version: str


class TextRecognizer(Protocol):
    """The contract every image-to-text model must follow."""
    version: str

    def predict(self, image: Image.Image) -> Prediction: ...

    def predict_batch(self, images: list[Image.Image]) -> list[Prediction]: ...


class TrOCRRecognizer:
    def __init__(
        self,
        model: VisionEncoderDecoderModel,
        processor: TrOCRProcessor,
        version: str,
        device: str | None = None,
        num_beams: int = 4,
        max_new_tokens: int = 32,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()   # eval() = switch off training behaviour (e.g. dropout)
        self.processor = processor
        self.version = version
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens        # charger IDs are short; this also caps latency

    @classmethod
    def from_pretrained(cls, name_or_path: str = DEFAULT_MODEL_NAME, version: str | None = None, **kwargs) -> "TrOCRRecognizer":
        """Load a model by Hugging Face name (downloads it the first time and
        caches it) or from a local folder containing a fine-tuned model."""
        processor = TrOCRProcessor.from_pretrained(name_or_path)
        model = VisionEncoderDecoderModel.from_pretrained(name_or_path)
        return cls(model, processor, version=version or name_or_path, **kwargs)

    def predict(self, image: Image.Image) -> Prediction:
        return self.predict_batch([image])[0]

    def predict_batch(self, images: list[Image.Image]) -> list[Prediction]:
        # 1. Pre-process: resize to the model's input size and normalise pixel values
        rgb_images = [img.convert("RGB") for img in images]
        pixel_values = self.processor(images=rgb_images, return_tensors="pt").pixel_values.to(self.device)

        # 2. Generate text. inference_mode() = no gradient bookkeeping -> faster, less memory
        with torch.inference_mode():
            output = self.model.generate(
                pixel_values,
                num_beams=self.num_beams,          # beam search: keep the 4 best partial guesses
                max_new_tokens=self.max_new_tokens,
                output_scores=True,                # we need the scores to compute confidence
                return_dict_in_generate=True,
            )

        texts = self.processor.batch_decode(output.sequences, skip_special_tokens=True)
        confidences = self._confidences(output)
        return [
            Prediction(text=text.strip(), confidence=conf, model_version=self.version)
            for text, conf in zip(texts, confidences)
        ]

    def _confidences(self, output) -> list[float]:
        """Confidence = geometric mean of the probability of every generated token.

        Example: tokens with probabilities 0.99, 0.98, 0.40 -> about 0.73.
        One uncertain character pulls the score down, which is what we want:
        a single wrong character means the wrong charger.
        """
        log_probs = self.model.compute_transition_scores(
            output.sequences,
            output.scores,
            getattr(output, "beam_indices", None),
            normalize_logits=self.num_beams == 1,  # beam search scores are already log-probabilities
        )
        # Ignore padding added after a sequence finished early in a batch
        generated = output.sequences[:, 1:]  # first token is the decoder start token
        pad_id = self.model.config.pad_token_id
        mask = generated[:, : log_probs.shape[1]] != pad_id

        result = []
        for row_scores, row_mask in zip(log_probs, mask):
            valid = row_scores[row_mask & torch.isfinite(row_scores)]
            result.append(float(valid.mean().exp()) if valid.numel() else 0.0)
        return result
