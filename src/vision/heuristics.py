from __future__ import annotations

import numpy as np
from PIL import Image

from .preprocess import to_rgb_image


def _resize_for_stats(image: Image.Image, max_edge: int = 384) -> np.ndarray:
    rgb = to_rgb_image(image)
    width, height = rgb.size
    longest = max(width, height)
    if longest > max_edge:
        scale = max_edge / longest
        rgb = rgb.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BILINEAR,
        )
    return np.asarray(rgb, dtype=np.float32)


def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, 1, mode="edge")
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = (
        kernel_x[0, 0] * padded[:-2, :-2]
        + kernel_x[0, 1] * padded[:-2, 1:-1]
        + kernel_x[0, 2] * padded[:-2, 2:]
        + kernel_x[1, 0] * padded[1:-1, :-2]
        + kernel_x[1, 1] * padded[1:-1, 1:-1]
        + kernel_x[1, 2] * padded[1:-1, 2:]
        + kernel_x[2, 0] * padded[2:, :-2]
        + kernel_x[2, 1] * padded[2:, 1:-1]
        + kernel_x[2, 2] * padded[2:, 2:]
    )
    gy = (
        kernel_y[0, 0] * padded[:-2, :-2]
        + kernel_y[0, 1] * padded[:-2, 1:-1]
        + kernel_y[0, 2] * padded[:-2, 2:]
        + kernel_y[1, 0] * padded[1:-1, :-2]
        + kernel_y[1, 1] * padded[1:-1, 1:-1]
        + kernel_y[1, 2] * padded[1:-1, 2:]
        + kernel_y[2, 0] * padded[2:, :-2]
        + kernel_y[2, 1] * padded[2:, 1:-1]
        + kernel_y[2, 2] * padded[2:, 2:]
    )
    return np.sqrt(gx * gx + gy * gy)


def _grid_energy(gray: np.ndarray, sat: np.ndarray) -> np.ndarray:
    energy = (gray - gray.mean()) ** 2 + sat * 80.0
    height, width = energy.shape
    rows = np.array_split(energy, 3, axis=0)
    cells = []
    for row in rows:
        cells.extend(block.mean() for block in np.array_split(row, 3, axis=1) if block.size)
    values = np.array(cells, dtype=np.float32)
    if values.size < 9:
        values = np.pad(values, (0, 9 - values.size))
    return values[:9]


def analyze_appearance(image: Image.Image) -> dict[str, str]:
    pixels = _resize_for_stats(image)
    if pixels.size == 0:
        return {
            "color": "无法读取色彩。",
            "composition": "无法读取构图。",
            "linework": "无法读取线条。",
        }

    rgb = pixels / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
    brightness = float(gray.mean())
    contrast = float(gray.std())

    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    sat = np.where(maxc > 1e-5, (maxc - minc) / np.clip(maxc, 1e-5, None), 0.0)
    saturation = float(sat.mean())

    # 简易色相：用 atan2 估计暖/冷占比
    hue = (np.degrees(np.arctan2(1.7320508 * (g - b), 2 * r - g - b)) + 360.0) % 360.0
    strong = sat > 0.12
    if np.any(strong):
        hues = hue[strong]
        warm = float(np.mean(((hues < 60) | (hues > 300))))
        cool = float(np.mean((hues > 90) & (hues < 270)))
    else:
        warm = cool = 0.0

    mean_rgb = rgb.mean(axis=(0, 1))
    dominant = f"RGB({int(mean_rgb[0] * 255)}, {int(mean_rgb[1] * 255)}, {int(mean_rgb[2] * 255)})"

    if saturation < 0.08:
        chroma = "接近单色或低饱和"
    elif saturation > 0.35:
        chroma = "高饱和、色彩鲜明"
    else:
        chroma = "中等饱和"

    if warm > cool + 0.12:
        temperature = "暖色倾向"
    elif cool > warm + 0.12:
        temperature = "冷色倾向"
    else:
        temperature = "冷暖较均衡"

    if brightness < 0.28:
        light = "画面偏暗"
    elif brightness > 0.72:
        light = "画面偏亮"
    else:
        light = "明暗适中"

    contrast_text = "高对比" if contrast > 0.22 else ("低对比" if contrast < 0.12 else "中等对比")

    energy = _grid_energy(gray, sat)
    center = float(energy[4])
    thirds = float(max(energy[0], energy[2], energy[6], energy[8], energy[1], energy[3], energy[5], energy[7]))
    if center > thirds * 1.18:
        layout = "主体更靠近画面中心"
    elif thirds > center * 1.08:
        layout = "主体偏向三分点或边角，中心较空"
    else:
        layout = "主体分布比较均衡"

    edges = _sobel_magnitude(gray)
    edge_mean = float(edges.mean())
    if edge_mean > 0.35:
        lines = "边缘密、线条或细节偏多"
    elif edge_mean < 0.12:
        lines = "边缘少，块面或柔焦更明显"
    else:
        lines = "线条密度中等"

    return {
        "color": f"{temperature}，{chroma}，{light}，{contrast_text}；平均色 {dominant}。",
        "composition": f"{layout}。亮度 {brightness:.2f}，对比 {contrast:.2f}。",
        "linework": f"{lines}（边缘强度 {edge_mean:.2f}）。",
        "brightness": f"{brightness:.3f}",
        "contrast": f"{contrast:.3f}",
        "saturation": f"{saturation:.3f}",
        "edge_mean": f"{edge_mean:.3f}",
    }
