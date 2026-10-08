"""
A tiny TrOCR, built from scratch, for tests and demos.

It has the same architecture as microsoft/trocr-base-printed (a ViT image
encoder + a TrOCR text decoder) but is ~2,000x smaller and starts with
random weights. So it:
  * needs no download (works offline, in CI, on any laptop)
  * trains in minutes on a CPU
  * runs through exactly the same training and inference code

It cannot read anything before training. That makes it a good demo: if
training works, you see the error rate go from ~100% to much lower.

The real model is used the same way; only the name changes:
    load_trocr("microsoft/trocr-base-printed")
"""

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import (
    PreTrainedTokenizerFast,
    TrOCRConfig,
    TrOCRForCausalLM,
    TrOCRProcessor,
    ViTConfig,
    ViTImageProcessor,
    ViTModel,
    VisionEncoderDecoderModel,
)

CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*-"

# Same special-token numbers as the real TrOCR: <s>=0, <pad>=1, </s>=2,
# and generation starts with token 2 (we checked microsoft/trocr-base-printed).
BOS, PAD, EOS, UNK = 0, 1, 2, 3


def build_tokenizer() -> PreTrainedTokenizerFast:
    """One token per character. Labels become "<s> N L * T N M ... </s>"."""
    vocab = {"<s>": BOS, "<pad>": PAD, "</s>": EOS, "<unk>": UNK}
    vocab.update({ch: i + 4 for i, ch in enumerate(CHARACTERS)})
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")   # split into characters
    backend.post_processor = processors.TemplateProcessing(                 # add <s> ... </s>
        single="<s> $A </s>", special_tokens=[("<s>", BOS), ("</s>", EOS)]
    )
    backend.decoder = decoders.Fuse()                                       # join without spaces
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<s>", pad_token="<pad>", eos_token="</s>", unk_token="<unk>"
    )


def build_tiny_trocr(height: int = 32, width: int = 128, hidden: int = 128, layers: int = 2,
                     patch: int | tuple[int, int] = (32, 4)):
    """Returns (model, processor) with random weights.

    The image is cut into thin vertical slices (32 x 4 pixels) instead of
    the usual 8 x 8 squares, so the encoder sees the text line as a row of
    32 columns read left to right, like classic OCR models (CRNN) do. With
    square patches a model starting from zero kept guessing an "average" ID
    instead of reading; with column slices it learned to read in ~30
    epochs on 900 crops. The real TrOCR doesn't need this trick: it was
    pre-trained on millions of text images and already knows how to read.
    """
    tokenizer = build_tokenizer()
    image_processor = ViTImageProcessor(size={"height": height, "width": width})
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

    encoder = ViTModel(ViTConfig(
        image_size=(height, width), patch_size=patch, hidden_size=hidden, num_hidden_layers=layers,
        num_attention_heads=4, intermediate_size=hidden * 2,
    ))
    decoder = TrOCRForCausalLM(TrOCRConfig(
        vocab_size=len(tokenizer), d_model=hidden, decoder_layers=layers, decoder_attention_heads=4,
        decoder_ffn_dim=hidden * 2, max_position_embeddings=64,
        pad_token_id=PAD, bos_token_id=BOS, eos_token_id=EOS, decoder_start_token_id=EOS,
    ))
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    for config in (model.config, model.generation_config):
        config.decoder_start_token_id = EOS
        config.pad_token_id = PAD
        config.eos_token_id = EOS
        config.bos_token_id = BOS
    return model, processor
