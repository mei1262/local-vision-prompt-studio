"""视觉分析：CLIP 零样本标签 + 色彩/构图规则。"""

from __future__ import annotations

from typing import Any

from PIL import Image

from io import BytesIO

from .heuristics import analyze_appearance
from .paths import load_llm_config, load_vision_config
from .runtime import LoadedBackend, encode_image, load_backend
from .resolve import infer_media_mode, pick_clip_scene, resolve_conflicts, scene_facts, scene_sentence
from .taxonomy import (
    build_content_text,
    format_group,
    format_top_candidates,
    score_tags,
    style_tag_labels,
)

_backend: LoadedBackend | None = None
_backend_error: str | None = None


def _as_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    return Image.fromarray(image)


def get_backend() -> LoadedBackend:
    global _backend, _backend_error
    if _backend is not None:
        return _backend
    if _backend_error is not None:
        raise RuntimeError(_backend_error)
    try:
        _backend = load_backend()
        return _backend
    except Exception as exc:  # noqa: BLE001
        _backend_error = str(exc)
        raise


def _jpeg_bytes(image: Image.Image) -> bytes:
    rgb = image.convert("RGB")
    rgb.thumbnail((768, 768))
    buffer = BytesIO()
    rgb.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def _vlm_prompt(extra_instruction: str, target_model: str) -> str:
    # 粗略意图不送给 moondream：它会跟着用户那几个字跑飞。
    _ = extra_instruction
    if str(target_model).startswith("视频"):
        return "Describe this image in two short English sentences. Add one camera move."
    return "Describe this image in two short English sentences."


def analyze_image(
    image: Any,
    extra_instruction: str = "",
    target_model: str = "",
    use_vlm: bool | None = None,
) -> dict[str, Any]:
    """对输入图片做视觉分析。无 ONNX 时仍返回色彩/构图规则结果。"""
    pil = _as_pil(image)
    appearance = analyze_appearance(pil)
    grouped: dict[str, list[tuple[str, float]]] = {}
    backend_note = ""

    try:
        backend = get_backend()
        config = load_vision_config()
        image_vec = encode_image(backend.session, pil, int(config.get("image_size", 224)))
        scores = (image_vec @ backend.tag_vectors.T).reshape(-1)
        score_list = scores.tolist()
        grouped = resolve_conflicts(score_tags(backend.tag_ids, score_list))
        top_text = format_top_candidates(backend.tag_ids, score_list)
        backend_note = (
            f"视觉后端：{backend.model_label} / {backend.provider}。"
            f" 候选相似度（不是准确率）：{top_text}。"
            " 分数只表示和词表的相对接近程度，请对照原图判断对不对。"
        )
        clip_content = build_content_text(grouped)
        content = clip_content
        style_tags = style_tag_labels(grouped)
        clip_scene = pick_clip_scene(image_vec, backend)
    except Exception as exc:  # noqa: BLE001
        clip_content = ""
        content = "尚未完成语义标签分析（规则分析仍可用）。"
        style_tags = []
        backend_note = str(exc)
        clip_scene = {}

    composition_tags = format_group(grouped, "composition")
    lighting_tags = format_group(grouped, "lighting")
    color_tags = format_group(grouped, "color")

    composition = appearance["composition"]
    if composition_tags:
        composition = f"{composition} 标签：{composition_tags}。"
    if lighting_tags:
        composition = f"{composition} 光线：{lighting_tags}。"

    color = appearance["color"]
    if color_tags:
        color = f"{color} 标签：{color_tags}。"

    caption = ""
    caption_note = ""
    caption_trusted = False
    caption_verdict = "描述与标签都来自 CLIP，不是另一个模型"
    mode = infer_media_mode(grouped, appearance)
    if clip_scene:
        caption = str(clip_scene["text"])
        caption_trusted = True
        caption_verdict = (
            "CLIP 场面检索 "
            f"{clip_scene['score']:.3f}；候选："
            + " / ".join(
                f"{item['text']} {item['score']:.3f}"
                for item in clip_scene.get("candidates") or []
                if isinstance(item, dict)
            )
        )
        caption_note = "场面句由 CLIP 在固定句库里选出"
        if clip_scene["mode"] in {"photo", "anime", "paint", "graphic"} and clip_scene["margin"] >= 0.003:
            mode = str(clip_scene["mode"])
    facts = scene_facts(grouped, mode)
    sentence = caption or scene_sentence(facts, mode)
    width, height = pil.size

    analysis = {
        "content": content,
        "clip_content": clip_content,
        "caption": caption,
        "caption_trusted": caption_trusted,
        "caption_verdict": caption_verdict,
        "vlm_text": "",
        "vlm_note": "未调用 moondream（开关关闭）",
        "vlm_attempted": False,
        "style_tags": style_tags,
        "composition": composition,
        "color": color,
        "linework": appearance["linework"],
        "notes": (
            f"{backend_note} {caption_note}。"
            f" 媒介判断：{mode}。场面句：{sentence}。"
        ),
        "grouped_tags": grouped,
        "media_mode": mode,
        "scene_sentence": sentence,
        "scene_candidates": list(clip_scene.get("candidates") or []) if clip_scene else [],
        "scene_facts": facts,
        "image_width": width,
        "image_height": height,
        "appearance": appearance,
    }
    want_vlm = use_vlm if use_vlm is not None else bool(load_llm_config().get("vlm_enabled"))
    if want_vlm:
        return attach_vlm(analysis, pil, extra_instruction, target_model)
    return analysis


def attach_vlm(
    analysis: dict[str, Any],
    image: Any,
    extra_instruction: str = "",
    target_model: str = "",
) -> dict[str, Any]:
    """在已有 CLIP 结果上只跑 moondream，不重算标签。"""
    from src.llm.ollama_client import describe_image, is_available, pick_vlm

    pil = _as_pil(image)
    analysis["vlm_attempted"] = True
    if not is_available():
        analysis["vlm_note"] = "Ollama 未运行，无法调用 moondream"
        return analysis
    vlm_name = pick_vlm()
    if not vlm_name:
        analysis["vlm_note"] = "没有 moondream。可执行：ollama pull moondream"
        return analysis
    text = describe_image( _jpeg_bytes(pil), _vlm_prompt(extra_instruction, target_model))
    analysis["vlm_text"] = text
    analysis["vlm_note"] = (
        f"小 VLM：{vlm_name}。"
        + ("已写出画面补充。" if text else "输出过短或乱码，已丢弃，仍用 CLIP。")
    )
    notes = str(analysis.get("notes") or "")
    analysis["notes"] = f"{notes} {analysis['vlm_note']}"
    return analysis
