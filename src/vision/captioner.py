"""轻量 ONNX 图像描述：BLIP-base，失败则 ViT-GPT2。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .hf_download import resolve_endpoints
from .paths import repo_root, resolve_path
from .preprocess import adapt_layout, preprocess_caption
from .runtime import available_providers, create_session

_captioner: "CaptionBackend | None" = None
_caption_error: str | None = None


def load_caption_config() -> dict:
    import json

    path = repo_root() / "configs" / "caption.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _preprocess(image: Image.Image, image_size: int) -> np.ndarray:
    return preprocess_caption(image, image_size=image_size)


def _encoder_output(session, pixels: np.ndarray) -> np.ndarray:
    feed = {session.get_inputs()[0].name: adapt_layout(pixels, session.get_inputs()[0].shape)}
    # 若还有 mask
    for item in session.get_inputs()[1:]:
        if "mask" in item.name.lower():
            feed[item.name] = np.ones((pixels.shape[0], 1, pixels.shape[2], pixels.shape[3]), dtype=np.int64)
    outputs = session.run(None, feed)
    hidden = np.asarray(outputs[0])
    if hidden.ndim == 2:
        hidden = hidden[:, None, :]
    return hidden.astype(np.float32, copy=False)


def _decoder_feeds(session, input_ids: np.ndarray, encoder_hidden: np.ndarray) -> dict[str, np.ndarray]:
    feeds: dict[str, np.ndarray] = {}
    enc_mask = np.ones(encoder_hidden.shape[:2], dtype=np.int64)
    dec_mask = np.ones_like(input_ids, dtype=np.int64)
    for item in session.get_inputs():
        name = item.name.lower()
        if "past" in name or "cache" in name:
            # merged 模型可能带空 past；用零占位会炸，遇到则改走无 past 输入失败
            raise RuntimeError(f"解码器需要 KV cache 输入：{item.name}")
        if "encoder_hidden" in name or name.endswith("encoder_outputs"):
            feeds[item.name] = encoder_hidden
        elif "encoder" in name and "mask" in name:
            feeds[item.name] = enc_mask
        elif "mask" in name:
            feeds[item.name] = dec_mask
        elif "input_ids" in name or "decoder_input" in name:
            feeds[item.name] = input_ids
        elif len(item.shape) == 3:
            feeds[item.name] = encoder_hidden
        else:
            feeds[item.name] = input_ids
    return feeds


def _next_token(logits: np.ndarray) -> int:
    if logits.ndim == 3:
        logits = logits[0, -1]
    elif logits.ndim == 2:
        logits = logits[0]
    return int(np.argmax(logits))


def _decode_tokens(tokenizer_path: Path, token_ids: list[int]) -> str:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return " ".join(text.replace("[SEP]", " ").replace("[CLS]", " ").split()).strip(" .")


@dataclass
class CaptionBackend:
    model_id: str
    model_label: str
    provider: str
    encoder: object
    decoder: object
    tokenizer_path: Path
    image_size: int
    bos_id: int
    eos_id: int
    max_new_tokens: int


def _try_load(model: dict, provider: str, max_new_tokens: int) -> CaptionBackend:
    encoder_path = resolve_path(model["encoder"])
    decoder_path = resolve_path(model["decoder"])
    tokenizer_path = resolve_path(model["tokenizer"])
    if not encoder_path.is_file() or not decoder_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError(f"缺少 caption 文件：{encoder_path.name} / {decoder_path.name}")
    encoder = create_session(encoder_path, provider)
    decoder = create_session(decoder_path, provider)
    dummy = np.zeros((1, 3, int(model["image_size"]), int(model["image_size"])), dtype=np.float32)
    hidden = _encoder_output(encoder, dummy)
    start = np.array([[int(model["bos_id"])]], dtype=np.int64)
    decoder.run(None, _decoder_feeds(decoder, start, hidden))
    return CaptionBackend(
        model_id=model["id"],
        model_label=model["label"],
        provider=provider,
        encoder=encoder,
        decoder=decoder,
        tokenizer_path=tokenizer_path,
        image_size=int(model["image_size"]),
        bos_id=int(model["bos_id"]),
        eos_id=int(model["eos_id"]),
        max_new_tokens=max_new_tokens,
    )


def load_captioner() -> CaptionBackend:
    global _captioner, _caption_error
    if _captioner is not None:
        return _captioner
    if _caption_error is not None:
        raise RuntimeError(_caption_error)

    config = load_caption_config()
    providers = [name for name in config.get("provider_order") or [] if name in available_providers()]
    errors: list[str] = []
    for model in config["models"]:
        for provider in providers:
            try:
                _captioner = _try_load(model, provider, int(config.get("max_new_tokens", 32)))
                return _captioner
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{model['id']} + {provider}：{exc}")
    _caption_error = "caption 模型未就绪。请运行 python scripts/prepare_caption_assets.py --source mirror --no-proxy"
    raise RuntimeError(_caption_error + " | " + " → ".join(errors[:6]))


def caption_image(image: Image.Image) -> tuple[str, str]:
    backend = load_captioner()
    pixels = _preprocess(image, backend.image_size)
    hidden = _encoder_output(backend.encoder, pixels)
    tokens = [backend.bos_id]
    for _ in range(backend.max_new_tokens):
        ids = np.array([tokens], dtype=np.int64)
        outputs = backend.decoder.run(None, _decoder_feeds(backend.decoder, ids, hidden))
        nxt = _next_token(np.asarray(outputs[0]))
        if nxt == backend.eos_id and len(tokens) > 1:
            break
        tokens.append(nxt)
        if tokens.count(nxt) >= 8:
            break
    text = _decode_tokens(backend.tokenizer_path, tokens)
    note = f"描述模型：{backend.model_label} / {backend.provider}"
    return text, note
