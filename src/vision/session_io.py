from __future__ import annotations

import numpy as np

from .preprocess import adapt_layout

# CLIP 词表中 "a photo" 的常见编码，用作双塔模型的占位文本。
DUMMY_TEXT_IDS = np.array(
    [[49406, 320, 1125, 49407] + [0] * 73],
    dtype=np.int64,
)


def pick_output(outputs: list, session, prefer: tuple[str, ...]) -> np.ndarray:
    for output, meta in zip(outputs, session.get_outputs()):
        name = meta.name.lower()
        if any(token in name for token in prefer):
            return np.asarray(output)
    return np.asarray(outputs[0])


def build_feeds(
    session,
    pixel_nchw: np.ndarray | None = None,
    input_ids: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    feeds: dict[str, np.ndarray] = {}
    for item in session.get_inputs():
        name = item.name
        lowered = name.lower()
        if "mask" in lowered:
            ids = input_ids if input_ids is not None else DUMMY_TEXT_IDS
            feeds[name] = (ids != 0).astype(np.int64)
        elif "input_ids" in lowered or "text" in lowered:
            feeds[name] = input_ids if input_ids is not None else DUMMY_TEXT_IDS
        else:
            if pixel_nchw is None:
                raise RuntimeError(f"会话需要图像输入 {name}，但未提供。")
            feeds[name] = adapt_layout(pixel_nchw, item.shape)
    return feeds


def dummy_pixels(image_size: int) -> np.ndarray:
    return np.zeros((1, 3, image_size, image_size), dtype=np.float32)
