from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import load_vision_config, resolve_path
from .preprocess import preprocess_clip
from .session_io import build_feeds, dummy_pixels, pick_output


def _import_ort():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "未安装 onnxruntime。请执行 pip install -r requirements.txt（含 onnxruntime-directml）。"
        ) from exc
    return ort


def available_providers() -> list[str]:
    ort = _import_ort()
    return list(ort.get_available_providers())


def create_session(model_path: Path, provider: str):
    ort = _import_ort()
    if provider not in ort.get_available_providers():
        raise RuntimeError(f"当前 ONNX Runtime 不支持 {provider}。")

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=[provider],
    )


def pool_embedding(raw: np.ndarray) -> np.ndarray:
    if raw.ndim == 3:
        raw = raw[:, 0, :]
    elif raw.ndim == 4:
        raw = raw.mean(axis=(2, 3))
    if raw.ndim != 2:
        raise RuntimeError(f"无法解析视觉输出形状：{raw.shape}")
    return raw.astype(np.float32, copy=False)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-8, None)


def warmup_session(session, image_size: int) -> None:
    pixels = dummy_pixels(image_size)
    session.run(None, build_feeds(session, pixel_nchw=pixels))


def encode_image(session, image, image_size: int) -> np.ndarray:
    nchw = preprocess_clip(image, image_size=image_size)
    outputs = session.run(None, build_feeds(session, pixel_nchw=nchw))
    picked = pick_output(outputs, session, prefer=("image_embed", "image_embeds", "pool"))
    return l2_normalize(pool_embedding(picked))


def probe_image_dim(session, image_size: int) -> int:
    outputs = session.run(None, build_feeds(session, pixel_nchw=dummy_pixels(image_size)))
    picked = pick_output(outputs, session, prefer=("image_embed", "image_embeds", "pool"))
    return int(l2_normalize(pool_embedding(picked)).shape[-1])


@dataclass
class LoadedBackend:
    model_id: str
    model_label: str
    provider: str
    session: object
    tag_ids: list[str]
    tag_vectors: np.ndarray
    scene_ids: list[str]
    scene_texts: list[str]
    scene_modes: list[str]
    scene_vectors: np.ndarray | None
    attempts: list[str]


def _load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=True)
    tag_ids = [str(item) for item in data["ids"].tolist()]
    vectors = l2_normalize(np.asarray(data["vectors"], dtype=np.float32))
    return tag_ids, vectors


def _candidate_onnx(model: dict) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    vision_path = resolve_path(model["vision_onnx"])
    files.append(("vision", vision_path))
    joint_key = model.get("joint_onnx")
    if joint_key:
        files.append(("joint", resolve_path(joint_key)))
    return files


def load_backend() -> LoadedBackend:
    """按 ViT-L + DirectML → ViT-L + CPU → ViT-B + DirectML → ViT-B + CPU 尝试加载。"""
    config = load_vision_config()
    image_size = int(config.get("image_size", 224))
    warmup = bool(config.get("warmup", True))
    provider_order = list(
        config.get("provider_order") or ["DmlExecutionProvider", "CPUExecutionProvider"]
    )
    attempts: list[str] = []
    last_error: Exception | None = None

    try:
        installed = available_providers()
        attempts.append("可用执行器：" + ", ".join(installed))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(str(exc)) from exc

    for model in config["models"]:
        embed_path = resolve_path(model["embeddings"])
        label = model.get("label", model["id"])
        if not embed_path.is_file():
            attempts.append(f"{label}：缺少标签向量 {embed_path}，请先运行 prepare 脚本")
            continue

        try:
            tag_ids, tag_vectors = _load_embeddings(embed_path)
        except Exception as exc:  # noqa: BLE001
            attempts.append(f"{label}：标签向量读取失败（{exc}）")
            last_error = exc
            continue

        scene_ids: list[str] = []
        scene_texts: list[str] = []
        scene_modes: list[str] = []
        scene_vectors = None
        scene_path = resolve_path(model["scene_embeddings"]) if model.get("scene_embeddings") else None
        if scene_path and scene_path.is_file():
            try:
                scene_data = np.load(scene_path, allow_pickle=True)
                scene_ids = [str(item) for item in scene_data["ids"].tolist()]
                scene_texts = [str(item) for item in scene_data["texts"].tolist()]
                scene_modes = [str(item) for item in scene_data["modes"].tolist()]
                scene_vectors = l2_normalize(np.asarray(scene_data["vectors"], dtype=np.float32))
            except Exception as exc:  # noqa: BLE001
                attempts.append(f"{label}：场面向量读取失败（{exc}）")

        for provider in provider_order:
            if provider not in installed:
                attempts.append(f"{label} + {provider}：运行时未提供该执行器，跳过")
                continue

            for kind, onnx_path in _candidate_onnx(model):
                if not onnx_path.is_file():
                    attempts.append(f"{label} {kind}：缺少 {onnx_path}")
                    continue
                try:
                    session = create_session(onnx_path, provider)
                    if warmup:
                        warmup_session(session, image_size)
                    dim = probe_image_dim(session, image_size)
                    if dim != tag_vectors.shape[-1]:
                        raise RuntimeError(
                            f"{kind} 视觉维度 {dim} 与标签向量 {tag_vectors.shape[-1]} 不一致"
                        )
                    attempts.append(f"{label} + {provider} + {kind}：成功")
                    return LoadedBackend(
                        model_id=model["id"],
                        model_label=label,
                        provider=provider,
                        session=session,
                        tag_ids=tag_ids,
                        tag_vectors=tag_vectors,
                        scene_ids=scene_ids,
                        scene_texts=scene_texts,
                        scene_modes=scene_modes,
                        scene_vectors=scene_vectors,
                        attempts=attempts,
                    )
                except Exception as exc:  # noqa: BLE001
                    attempts.append(f"{label} + {provider} + {kind}：失败（{exc}）")
                    last_error = exc

    detail = " | ".join(attempts)
    if last_error is not None:
        raise RuntimeError(f"视觉后端全部失败。{detail}") from last_error
    raise RuntimeError(
        "视觉后端未就绪。请先运行 python scripts/prepare_vision_assets.py。"
        f" 详情：{detail}"
    )
