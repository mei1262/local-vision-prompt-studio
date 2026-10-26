from __future__ import annotations

from pathlib import Path

import numpy as np

from .paths import load_scene_bank, load_tag_vocab
from .runtime import l2_normalize, pool_embedding
from .session_io import build_feeds, dummy_pixels, pick_output
from .taxonomy import flatten_tags


def _tokenizer(tokenizer_path: Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tokenizer_path))


def tokenize_clip(tokenizer, text: str, context_length: int = 77) -> np.ndarray:
    encoded = tokenizer.encode(text)
    ids = list(encoded.ids)
    if not ids or ids[0] != 49406:
        ids = [49406, *ids]
    if ids[-1] != 49407:
        ids.append(49407)
    ids = ids[:context_length]
    if len(ids) < context_length:
        ids = ids + [0] * (context_length - len(ids))
    return np.asarray(ids, dtype=np.int64)


def encode_texts(session, tokenizer, texts: list[str], image_size: int = 224) -> np.ndarray:
    vectors = []
    pixels = dummy_pixels(image_size)
    for text in texts:
        input_ids = tokenize_clip(tokenizer, text)[None, ...]
        outputs = session.run(
            None,
            build_feeds(session, pixel_nchw=pixels, input_ids=input_ids)
            if any("pixel" in item.name.lower() or "image" in item.name.lower() for item in session.get_inputs())
            else _text_only_feeds(session, input_ids),
        )
        picked = pick_output(outputs, session, prefer=("text_embed", "text_embeds", "embed", "pool"))
        if picked.ndim == 3:
            lengths = (input_ids != 0).sum(axis=-1) - 1
            picked = picked[np.arange(picked.shape[0]), np.clip(lengths, 0, picked.shape[1] - 1)]
        else:
            picked = pool_embedding(picked)
        vectors.append(np.asarray(picked[0], dtype=np.float32))
    return l2_normalize(np.stack(vectors, axis=0))


def _text_only_feeds(session, input_ids: np.ndarray) -> dict[str, np.ndarray]:
    feeds: dict[str, np.ndarray] = {}
    batch = input_ids if input_ids.ndim == 2 else input_ids[None, ...]
    for item in session.get_inputs():
        if "mask" in item.name.lower():
            feeds[item.name] = (batch != 0).astype(np.int64)
        else:
            feeds[item.name] = batch
    return feeds


def bake_tag_embeddings(
    text_session,
    tokenizer_path: Path,
    output_path: Path,
    image_size: int = 224,
) -> Path:
    vocab = load_tag_vocab()
    templates = list(vocab.get("templates") or ["{}"])
    tokenizer = _tokenizer(tokenizer_path)
    tags = flatten_tags()
    vectors = []
    for tag in tags:
        prompts = [template.format(tag["en"]) for template in templates]
        prompt_vecs = encode_texts(text_session, tokenizer, prompts, image_size=image_size)
        vectors.append(l2_normalize(prompt_vecs.mean(axis=0, keepdims=True))[0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        ids=np.asarray([tag["id"] for tag in tags], dtype=object),
        vectors=np.asarray(vectors, dtype=np.float32),
    )
    return output_path


def bake_scene_embeddings(
    text_session,
    tokenizer_path: Path,
    output_path: Path,
    image_size: int = 224,
) -> Path:
    bank = load_scene_bank()
    tokenizer = _tokenizer(tokenizer_path)
    scenes = list(bank.get("scenes") or [])
    texts = [str(item["en"]) for item in scenes]
    vectors = encode_texts(text_session, tokenizer, texts, image_size=image_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        ids=np.asarray([item["id"] for item in scenes], dtype=object),
        texts=np.asarray(texts, dtype=object),
        modes=np.asarray([item.get("mode", "general") for item in scenes], dtype=object),
        vectors=np.asarray(vectors, dtype=np.float32),
    )
    return output_path
