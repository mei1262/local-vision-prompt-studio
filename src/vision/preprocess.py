from __future__ import annotations

import numpy as np
from PIL import Image

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def to_rgb_image(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGB", image.size, (255, 255, 255))
        alpha = image.split()[-1]
        background.paste(image, mask=alpha)
        return background
    return image.convert("RGB")


CAPTION_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
CAPTION_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def preprocess_clip(image: Image.Image, image_size: int = 224) -> np.ndarray:
    """OpenAI CLIP 常用预处理：短边缩放、中心裁切、归一化，输出 NCHW float32。"""
    rgb = to_rgb_image(image)
    width, height = rgb.size
    short_side = min(width, height)
    if short_side != image_size:
        scale = image_size / short_side
        rgb = rgb.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BICUBIC,
        )
    width, height = rgb.size
    left = max(0, (width - image_size) // 2)
    top = max(0, (height - image_size) // 2)
    rgb = rgb.crop((left, top, left + image_size, top + image_size))

    pixels = np.asarray(rgb, dtype=np.float32) / 255.0
    pixels = (pixels - CLIP_MEAN) / CLIP_STD
    return np.transpose(pixels, (2, 0, 1))[None, ...]


def preprocess_caption(image: Image.Image, image_size: int = 224) -> np.ndarray:
    """ViT-GPT2 / BLIP 常用：直接缩放到方形，均值 0.5。"""
    rgb = to_rgb_image(image).resize((image_size, image_size), Image.Resampling.BICUBIC)
    pixels = np.asarray(rgb, dtype=np.float32) / 255.0
    pixels = (pixels - CAPTION_MEAN) / CAPTION_STD
    return np.transpose(pixels, (2, 0, 1))[None, ...]


def adapt_layout(nchw: np.ndarray, input_shape) -> np.ndarray:
    """按 ONNX 输入维序在 NCHW / NHWC 之间转换。"""
    if len(input_shape) == 4 and input_shape[-1] == 3:
        return np.transpose(nchw, (0, 2, 3, 1))
    return nchw
