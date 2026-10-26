"""Gradio 入口：视觉分析 + 提示词工作台。"""

from __future__ import annotations

import hashlib
from typing import Any

import gradio as gr
from PIL import Image

from src.prompt.generator import generate_prompts, generate_video_prompts
from src.prompt.video import suggest_slots
from src.prompt.video import (
    AMBIENCE_CHOICES,
    CAMERA_CHOICES,
    DURATION_CHOICES,
    FOCUS_CHOICES,
    GRADE_CHOICES,
    LIGHT_CHOICES,
    MOTION_CHOICES,
    SHOT_CHOICES,
    STYLE_CHOICES,
)
from src.vision.analyzer import analyze_image, attach_vlm
from src.workbench.locks import (
    character_lock_from_analysis,
    merge_locks,
    style_lock_from_analysis,
)
from src.workbench.store import (
    append_history,
    delete_history,
    delete_profile,
    find_history,
    find_profile,
    history_choices,
    profile_choices,
    selected_lock_texts,
    upsert_profile,
)

TARGET_MODELS = ["SDXL", "Flux"]
KIND_LABELS = {"character": "角色", "style": "风格"}
KIND_CHOICES = ["角色", "风格"]


def _image_key(image: Any) -> str:
    raw = image.tobytes() if hasattr(image, "tobytes") else bytes(str(getattr(image, "size", "")), "utf-8")
    return hashlib.sha1(raw[: 256 * 1024] + raw[-4096:]).hexdigest()


def _as_images(value: Any, limit: int) -> list[Image.Image]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    images: list[Image.Image] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            item = item[0]
        if item is None:
            continue
        path = getattr(item, "path", None) or getattr(item, "name", None)
        if isinstance(item, dict):
            path = item.get("path") or item.get("name") or path
            item = item.get("image") or item
        try:
            if isinstance(item, Image.Image):
                images.append(item.convert("RGB"))
            elif path:
                with Image.open(path) as opened:
                    images.append(opened.convert("RGB"))
        except (OSError, ValueError):
            continue
        if len(images) >= limit:
            break
    return images


def _candidate_label(item: dict[str, Any]) -> str:
    return f"{item['text']}  ({item['score']:.3f})"


def _candidate_labels(analysis: dict[str, Any]) -> list[str]:
    labels = []
    for item in analysis.get("scene_candidates") or []:
        if isinstance(item, dict) and item.get("text"):
            labels.append(_candidate_label(item))
    return labels


def _text_from_label(analysis: dict[str, Any], label: str) -> str:
    for item in analysis.get("scene_candidates") or []:
        if isinstance(item, dict) and _candidate_label(item) == label:
            return str(item.get("text") or "")
    return label


def _radio_state(analysis: dict[str, Any]) -> tuple[Any, str]:
    labels = _candidate_labels(analysis)
    scene = str(analysis.get("scene_sentence") or "")
    value = labels[0] if labels else None
    for item in analysis.get("scene_candidates") or []:
        if isinstance(item, dict) and item.get("text") == scene:
            value = _candidate_label(item)
            break
    return gr.update(choices=labels, value=value), scene


def _kind_value(label: str) -> str:
    return "style" if label in ("style", "风格") else "character"


def _kind_label(value: str) -> str:
    return KIND_LABELS.get(value, "角色")


def _copy_pack(positive: str, negative: str, params: str) -> str:
    return (
        "=== POSITIVE ===\n"
        f"{positive}\n\n"
        "=== NEGATIVE ===\n"
        f"{negative}\n\n"
        "=== PARAMS ===\n"
        f"{params}"
    )


def _format_outputs(analysis: dict[str, Any], result: dict[str, str]) -> tuple[str, str, str, str, str]:
    analysis_text = "\n".join(
        line
        for line in [
            (
                f"描述：{analysis.get('caption')}（{analysis.get('caption_verdict')}）"
                if analysis.get("caption")
                else f"描述：{analysis.get('caption_verdict') or '未启用'}"
            ),
            f"标签：{analysis.get('clip_content') or analysis.get('content', '')}",
            f"CLIP场面：{analysis.get('scene_sentence', '')}",
            f"身份/风格锁：{', '.join(analysis.get('lock_phrases') or []) or '（无）'}",
            f"VLM补充：{analysis.get('vlm_text') or analysis.get('vlm_note') or '（无）'}",
            f"媒介：{analysis.get('media_mode', '')}",
            f"风格标签：{', '.join(analysis.get('style_tags') or []) or '（空）'}",
            f"构图：{analysis.get('composition', '')}",
            f"色彩：{analysis.get('color', '')}",
            f"线条：{analysis.get('linework', '')}",
            f"备注：{analysis.get('notes', '')}",
            f"生成：{result.get('notes', '')}",
            analysis.get("lock_notes") or "",
        ]
        if line
    )
    positive = result.get("positive", "")
    negative = result.get("negative", "")
    params = result.get("params", "")
    return analysis_text, positive, negative, params, _copy_pack(positive, negative, params)


def _for_prompt(analysis: dict[str, Any], use_vlm: bool) -> dict[str, Any]:
    copied = dict(analysis)
    if not use_vlm:
        copied["vlm_text"] = ""
        copied["vlm_note"] = "未调用 moondream（开关关闭）"
    return copied


def _history_update(
    choices: list[str] | None = None,
    value: str | None = None,
    side: str = "image",
) -> Any:
    items = choices if choices is not None else history_choices(side)
    return gr.update(choices=items, value=value if items else None)


def _empty(cache: dict[str, Any] | None, message: str = "请先上传本镜图片。") -> tuple:
    return (
        message,
        "",
        "",
        "",
        "",
        cache or {},
        gr.update(choices=[], value=None),
        "",
        _history_update(),
        message,
    )


def _cached_analysis(image: Image.Image, cache: dict[str, Any], bucket: str) -> dict[str, Any]:
    extras = dict(cache.get("extras") or {})
    bucket_map = dict(extras.get(bucket) or {})
    key = _image_key(image)
    analysis = bucket_map.get(key)
    if analysis is None:
        analysis = analyze_image(image, use_vlm=False)
        bucket_map[key] = analysis
        extras[bucket] = bucket_map
        cache["extras"] = extras
    return analysis


def _apply_locks(
    analysis: dict[str, Any],
    cache: dict[str, Any],
    character_refs: Any,
    style_refs: Any,
    profile_names: list[str] | None,
) -> dict[str, Any]:
    character_analyses = [
        _cached_analysis(image, cache, "character") for image in _as_images(character_refs, 3)
    ]
    style_analyses = [
        _cached_analysis(image, cache, "style") for image in _as_images(style_refs, 2)
    ]
    return merge_locks(
        analysis,
        character_analyses=character_analyses,
        style_analyses=style_analyses,
        profile_texts=selected_lock_texts(profile_names),
    )


def _finish(
    analysis: dict[str, Any],
    target: str,
    extra: str,
    use_vlm: bool,
    cache: dict[str, Any],
    profile_names: list[str] | None,
) -> tuple:
    prompt_analysis = _for_prompt(analysis, bool(use_vlm))
    result = generate_prompts(
        analysis=prompt_analysis,
        target_model=target,
        extra_instruction=extra,
    )
    radio, scene = _radio_state(prompt_analysis)
    cache["analysis"] = prompt_analysis
    cache["last_result"] = {
        "side": "image",
        "target_model": target,
        "extra_instruction": extra,
        "scene_sentence": scene,
        "positive": result.get("positive", ""),
        "negative": result.get("negative", ""),
        "params": result.get("params", ""),
        "profile_names": list(profile_names or []),
        "lock_notes": prompt_analysis.get("lock_notes") or "",
    }
    choices = append_history(cache["last_result"])
    formatted = _format_outputs(prompt_analysis, result)
    return (
        *formatted,
        cache,
        radio,
        scene,
        _history_update(choices, choices[0] if choices else None),
        "已生成，并写入本地历史（只存提示，不存原图）。",
    )


def run_pipeline(
    image: Any,
    character_refs: Any,
    style_refs: Any,
    target_model: str,
    extra_instruction: str,
    use_vlm: bool,
    profile_names: list[str] | None,
    cache: dict[str, Any] | None,
) -> tuple:
    if image is None:
        return _empty(cache)

    target = target_model or TARGET_MODELS[0]
    extra = extra_instruction or ""
    cache = dict(cache or {})
    key = _image_key(image)
    analysis = cache.get("analysis") if cache.get("key") == key else None
    clip = dict(cache.get("clip") or {})

    if analysis is None:
        analysis = dict(clip[key]) if key in clip else analyze_image(
            image,
            extra_instruction=extra,
            target_model=target,
            use_vlm=bool(use_vlm),
        )
        if key not in clip:
            clip[key] = analysis
    elif use_vlm and not analysis.get("vlm_attempted"):
        analysis = attach_vlm(analysis, image, extra, target)
        clip[key] = {k: v for k, v in analysis.items() if k not in {"lock_phrases", "lock_notes"}}

    cache["key"] = key
    cache["clip"] = clip
    locked = _apply_locks(analysis, cache, character_refs, style_refs, profile_names)
    return _finish(locked, target, extra, use_vlm, cache, profile_names)


def apply_scene_sentence(
    scene_text: str,
    character_refs: Any,
    style_refs: Any,
    target_model: str,
    extra_instruction: str,
    use_vlm: bool,
    profile_names: list[str] | None,
    cache: dict[str, Any] | None,
) -> tuple:
    cache = dict(cache or {})
    analysis = cache.get("analysis")
    if not analysis:
        return _empty(cache, "请先点「分析并生成提示词」，再改场面句。")
    analysis = dict(analysis)
    sentence = (scene_text or "").strip()
    if sentence:
        analysis["scene_sentence"] = sentence
        analysis["caption"] = sentence
        for item in analysis.get("scene_candidates") or []:
            if isinstance(item, dict) and item.get("text") == sentence and item.get("mode"):
                analysis["media_mode"] = item["mode"]
                break
    locked = _apply_locks(analysis, cache, character_refs, style_refs, profile_names)
    target = target_model or TARGET_MODELS[0]
    return _finish(locked, target, extra_instruction or "", use_vlm, cache, profile_names)


def on_pick_candidate(label: str, cache: dict[str, Any] | None) -> str:
    analysis = (cache or {}).get("analysis") or {}
    return _text_from_label(analysis, label or "")


def on_vlm_toggle(
    use_vlm: bool,
    image: Any,
    character_refs: Any,
    style_refs: Any,
    target_model: str,
    extra_instruction: str,
    profile_names: list[str] | None,
    cache: dict[str, Any] | None,
) -> tuple:
    return run_pipeline(
        image,
        character_refs,
        style_refs,
        target_model,
        extra_instruction,
        bool(use_vlm),
        profile_names,
        cache,
    )


def save_current_history(cache: dict[str, Any] | None) -> tuple[Any, str]:
    last = (cache or {}).get("last_result")
    if not last or not last.get("positive"):
        return _history_update(), "没有可保存的提示。先分析一次。"
    choices = append_history(last)
    return _history_update(choices, choices[0] if choices else None), "已保存到历史。"


def load_history_entry(label: str, cache: dict[str, Any] | None) -> tuple:
    item = find_history(label or "")
    cache = dict(cache or {})
    if not item:
        return _empty(cache, "没有这条历史。")
    positive = str(item.get("positive") or "")
    negative = str(item.get("negative") or "")
    params = str(item.get("params") or "")
    scene = str(item.get("scene_sentence") or "")
    cache["last_result"] = {
        "target_model": item.get("target_model") or TARGET_MODELS[0],
        "extra_instruction": item.get("extra_instruction") or "",
        "scene_sentence": scene,
        "positive": positive,
        "negative": negative,
        "params": params,
        "profile_names": list(item.get("profile_names") or []),
        "lock_notes": item.get("lock_notes") or "",
    }
    analysis_text = (
        "已载入历史提示。这是上次写好的词，拿去出新图；"
        "不会自动重新上传原图。\n"
        f"场面句：{scene}\n"
        f"当时意图：{item.get('extra_instruction') or '（无）'}\n"
        f"档案：{', '.join(item.get('profile_names') or []) or '（无）'}\n"
        f"{item.get('lock_notes') or ''}"
    )
    target = item.get("target_model") or TARGET_MODELS[0]
    if target not in TARGET_MODELS:
        target = TARGET_MODELS[0]
    return (
        analysis_text,
        positive,
        negative,
        params,
        _copy_pack(positive, negative, params),
        cache,
        gr.update(),
        scene,
        _history_update(value=label),
        "已载入。可直接复制，或换一张本镜再生成。",
        gr.update(value=target),
        str(item.get("extra_instruction") or ""),
    )


def remove_history_entry(label: str) -> tuple[Any, str]:
    choices = delete_history(label or "", "image")
    return _history_update(choices, choices[0] if choices else None), "已删除该条历史。"


def fill_profile(name: str) -> tuple[str, str, str]:
    item = find_profile(name or "")
    if not item:
        return "角色", name or "", ""
    return (
        _kind_label(str(item.get("kind") or "character")),
        str(item.get("name") or ""),
        str(item.get("lock_text") or ""),
    )


def save_profile_entry(kind: str, name: str, lock_text: str) -> tuple[Any, Any, str]:
    names = upsert_profile(_kind_value(kind), name, lock_text)
    if not (name or "").strip() or not (lock_text or "").strip():
        return (
            gr.update(choices=names),
            gr.update(choices=names),
            "名称和锁词都要填。",
        )
    return (
        gr.update(choices=names),
        gr.update(choices=names, value=(name or "").strip()),
        f"已保存档案「{(name or '').strip()}」。生成新图时勾选即可套用。",
    )


def remove_profile_entry(name: str) -> tuple:
    names = delete_profile(name or "")
    return (
        gr.update(choices=names, value=[]),
        gr.update(choices=names, value=None),
        "角色",
        "",
        "",
        "已删除档案。" if name else "没有选中档案。",
    )


def extract_lock_draft(
    kind: str,
    image: Any,
    character_refs: Any,
    style_refs: Any,
    cache: dict[str, Any] | None,
) -> tuple[str, str]:
    cache = dict(cache or {})
    use_style = _kind_value(kind) == "style"
    refs = _as_images(style_refs if use_style else character_refs, 1)
    source = refs[0] if refs else image
    if source is None:
        analysis = (cache.get("analysis") or {}) if cache else {}
    else:
        analysis = _cached_analysis(source, cache, "style" if use_style else "character")
    if not analysis:
        return "", "没有可提取的图。上传本镜或对应参考图。"
    text = style_lock_from_analysis(analysis) if use_style else character_lock_from_analysis(analysis)
    if not text:
        return "", "CLIP 没抽出可用锁词，请手写档案。"
    hint = "风格" if use_style else "角色"
    return text, f"已从{hint}参考抽出草稿，请改成稳定外貌/画风后再保存。不要把构图写进去。"


def _switch_side(side: str) -> tuple[Any, Any]:
    image_on = side != "视频侧"
    return gr.update(visible=image_on), gr.update(visible=not image_on)


def _video_analysis(image: Any, cache: dict[str, Any]) -> dict[str, Any]:
    key = _image_key(image)
    clip = dict(cache.get("clip") or {})
    if key in clip:
        analysis = dict(clip[key])
    else:
        analysis = analyze_image(image, use_vlm=False)
        clip[key] = analysis
        cache["clip"] = clip
    analysis.pop("lock_phrases", None)
    analysis.pop("lock_notes", None)
    analysis["vlm_text"] = ""
    return analysis


def _video_empty(cache: dict[str, Any] | None, message: str) -> tuple:
    return (
        message,
        "",
        "",
        "",
        "",
        cache or {},
        gr.update(choices=[], value=None),
        "",
        _history_update(side="video"),
        message,
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )


def _finish_video(
    analysis: dict[str, Any],
    cache: dict[str, Any],
    style: str,
    camera: str,
    motion: str,
    duration: str,
    shot_size: str,
    light: str,
    focus: str,
    grade: str,
    ambience: str,
    environment: str,
    subject: str,
    music: str,
    action: str,
    extra: str,
) -> tuple:
    hinted = suggest_slots(analysis)
    environment = (environment or "").strip() or hinted["environment"]
    subject = (subject or "").strip() or hinted["subject"]
    music = (music or "").strip()
    action = (action or "").strip()
    result = generate_video_prompts(
        analysis,
        style=style or "通用",
        camera=camera or "缓慢推进",
        motion=motion or "缓慢",
        duration=duration or "5秒",
        shot_size=shot_size or "跟随场面句",
        light=light or "跟随场面句",
        focus=focus or "主体清晰",
        grade=grade or "清晰稳定",
        ambience=ambience or "不写",
        environment=environment,
        subject=subject,
        music=music,
        action=action,
        extra_instruction=extra,
    )
    radio, scene = _radio_state(analysis)
    cache["video_analysis"] = analysis
    cache["video_last"] = {
        "side": "video",
        "target_model": style or "通用",
        "extra_instruction": extra,
        "scene_sentence": scene,
        "positive": result.get("positive", ""),
        "negative": result.get("negative", ""),
        "params": result.get("params", ""),
        "profile_names": [],
        "lock_notes": "",
        "camera": camera,
        "motion": motion,
        "duration": duration,
        "shot_size": shot_size,
        "light": light,
        "focus": focus,
        "grade": grade,
        "ambience": ambience,
        "environment": environment,
        "subject": subject,
        "music": music,
        "action": action,
    }
    choices = append_history(cache["video_last"])
    analysis_text, positive, negative, params, pack = _format_outputs(analysis, result)
    note = str(result.get("notes") or "")
    if note and note not in analysis_text:
        analysis_text = f"{analysis_text}\n生成：{note}"
    return (
        analysis_text,
        positive,
        negative,
        params,
        pack,
        cache,
        radio,
        scene,
        _history_update(choices, choices[0] if choices else None, side="video"),
        "已按这一镜头的模板生成，并写入视频历史。参考图请另交视频模型。",
        environment,
        subject,
        music,
        action,
    )


def run_video(
    image: Any,
    style: str,
    shot_size: str,
    camera: str,
    light: str,
    focus: str,
    motion: str,
    grade: str,
    ambience: str,
    duration: str,
    environment: str,
    subject: str,
    music: str,
    action: str,
    extra_instruction: str,
    cache: dict[str, Any] | None,
) -> tuple:
    if image is None:
        return _video_empty(cache, "请先上传作为画面锁的静帧。")
    cache = dict(cache or {})
    analysis = _video_analysis(image, cache)
    cache["video_key"] = _image_key(image)
    return _finish_video(
        analysis,
        cache,
        style,
        camera,
        motion,
        duration,
        shot_size,
        light,
        focus,
        grade,
        ambience,
        environment,
        subject,
        music,
        action,
        extra_instruction or "",
    )


def apply_video_scene(
    scene_text: str,
    style: str,
    shot_size: str,
    camera: str,
    light: str,
    focus: str,
    motion: str,
    grade: str,
    ambience: str,
    duration: str,
    environment: str,
    subject: str,
    music: str,
    action: str,
    extra_instruction: str,
    cache: dict[str, Any] | None,
) -> tuple:
    cache = dict(cache or {})
    analysis = cache.get("video_analysis")
    if not analysis:
        return _video_empty(cache, "请先点「生成视频提示」，再改场面句。")
    analysis = dict(analysis)
    sentence = (scene_text or "").strip()
    if sentence:
        analysis["scene_sentence"] = sentence
        analysis["caption"] = sentence
        for item in analysis.get("scene_candidates") or []:
            if isinstance(item, dict) and item.get("text") == sentence and item.get("mode"):
                analysis["media_mode"] = item["mode"]
                break
    return _finish_video(
        analysis,
        cache,
        style,
        camera,
        motion,
        duration,
        shot_size,
        light,
        focus,
        grade,
        ambience,
        environment,
        subject,
        music,
        action,
        extra_instruction or "",
    )


def on_pick_video(label: str, cache: dict[str, Any] | None) -> str:
    analysis = (cache or {}).get("video_analysis") or {}
    return _text_from_label(analysis, label or "")


def save_video_history(cache: dict[str, Any] | None) -> tuple[Any, str]:
    last = (cache or {}).get("video_last")
    if not last or not last.get("positive"):
        return _history_update(side="video"), "没有可保存的视频提示。先生成一次。"
    choices = append_history(last)
    return _history_update(choices, choices[0] if choices else None, side="video"), "已保存到视频历史。"


def _video_choice(value: str, choices: list[str], default: str) -> Any:
    return gr.update(value=value if value in choices else default)


def load_video_history(label: str, cache: dict[str, Any] | None) -> tuple:
    item = find_history(label or "")
    cache = dict(cache or {})
    if not item or (item.get("side") or "image") != "video":
        return (
            *_video_empty(cache, "没有这条视频历史。"),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    positive = str(item.get("positive") or "")
    negative = str(item.get("negative") or "")
    params = str(item.get("params") or "")
    scene = str(item.get("scene_sentence") or "")
    cache["video_last"] = {
        "side": "video",
        "target_model": item.get("target_model") or "通用",
        "extra_instruction": item.get("extra_instruction") or "",
        "scene_sentence": scene,
        "positive": positive,
        "negative": negative,
        "params": params,
        "camera": item.get("camera") or "",
        "motion": item.get("motion") or "",
        "duration": item.get("duration") or "",
        "shot_size": item.get("shot_size") or "",
        "light": item.get("light") or "",
        "focus": item.get("focus") or "",
        "grade": item.get("grade") or "",
        "ambience": item.get("ambience") or "",
        "environment": item.get("environment") or "",
        "subject": item.get("subject") or "",
        "music": item.get("music") or "",
        "action": item.get("action") or "",
    }
    analysis_text = (
        "已载入视频历史。这是一条视频提示，不会重新分析原图。\n"
        f"场面句：{scene}\n"
        f"当时意图：{item.get('extra_instruction') or '（无）'}"
    )
    style = item.get("target_model") or "通用"
    if style not in STYLE_CHOICES:
        style = "通用"
    camera = item.get("camera") if item.get("camera") in CAMERA_CHOICES else CAMERA_CHOICES[1]
    motion = item.get("motion") if item.get("motion") in MOTION_CHOICES else MOTION_CHOICES[1]
    duration = item.get("duration") if item.get("duration") in DURATION_CHOICES else DURATION_CHOICES[1]
    return (
        analysis_text,
        positive,
        negative,
        params,
        _copy_pack(positive, negative, params),
        cache,
        gr.update(),
        scene,
        _history_update(side="video", value=label),
        "已载入视频提示。",
        str(item.get("environment") or ""),
        str(item.get("subject") or ""),
        str(item.get("music") or ""),
        str(item.get("action") or ""),
        gr.update(value=style),
        _video_choice(str(item.get("shot_size") or ""), SHOT_CHOICES, "跟随场面句"),
        gr.update(value=camera),
        _video_choice(str(item.get("light") or ""), LIGHT_CHOICES, "跟随场面句"),
        _video_choice(str(item.get("focus") or ""), FOCUS_CHOICES, "主体清晰"),
        gr.update(value=motion),
        _video_choice(str(item.get("grade") or ""), GRADE_CHOICES, "清晰稳定"),
        _video_choice(str(item.get("ambience") or ""), AMBIENCE_CHOICES, "不写"),
        gr.update(value=duration),
        str(item.get("extra_instruction") or ""),
    )


def remove_video_history(label: str) -> tuple[Any, str]:
    choices = delete_history(label or "", "video")
    return _history_update(choices, choices[0] if choices else None, side="video"), "已删除该条视频历史。"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Local Vision Prompt Studio") as demo:
        cache = gr.State({})
        gr.Markdown(
            """
            # Local Vision Prompt Studio
            本机只出提示词，不渲染。用下面的开关切换**图片侧**和**视频侧**。两边表单、历史分开。
            同一张图若刚分析过，另一侧再分析会复用 CLIP 结果，不重算。
            """
        )
        side = gr.Radio(["图片侧", "视频侧"], value="图片侧", label="功能")

        with gr.Column(visible=True) as image_panel:
            gr.Markdown(
                """
                **图片侧**：本镜决定新构图，复制给 SDXL / Flux 出新图。
                档案不用填。生成后自动写入图片历史；状态说明不是历史本身。
                """
            )
            with gr.Row():
                with gr.Column(scale=1):
                    image = gr.Image(label="本镜（新画面 / 构图）", type="pil")
                    character_refs = gr.Gallery(
                        label="角色参考（可选，最多 3 张：锁外貌，不锁姿势）",
                        type="pil",
                        columns=3,
                        height=140,
                        object_fit="contain",
                    )
                    style_refs = gr.Gallery(
                        label="风格参考（可选，最多 2 张：锁画风光线，不锁场面）",
                        type="pil",
                        columns=2,
                        height=140,
                        object_fit="contain",
                    )
                    target_model = gr.Dropdown(
                        label="目标生成模型",
                        choices=TARGET_MODELS,
                        value=TARGET_MODELS[0],
                    )
                    extra_instruction = gr.Textbox(
                        label="粗略意图（可选，只影响新图写法）",
                        placeholder="例如：更冷清一点 / 换成夜景 / 保留发型",
                        lines=3,
                    )
                    profile_apply = gr.CheckboxGroup(
                        label="本次套用的档案",
                        choices=profile_choices(),
                        value=[],
                    )
                    use_vlm = gr.Checkbox(
                        label="使用 moondream 看图补充（慢，可能乱码；默认关）",
                        value=False,
                    )
                    run_btn = gr.Button("分析并生成提示词", variant="primary")
                    scene_pick = gr.Radio(
                        label="CLIP 场面句候选（点选填入下方，再点生成）",
                        choices=[],
                        value=None,
                    )
                    scene_edit = gr.Textbox(
                        label="场面句（可编辑，这是新图的构图）",
                        lines=3,
                        placeholder="分析后会出现英文场面句，可改成你认为更准的一句",
                    )
                    apply_btn = gr.Button("按当前场面句生成")

                with gr.Column(scale=1):
                    analysis_out = gr.Textbox(label="视觉分析", lines=10, show_copy_button=True)
                    positive_out = gr.Textbox(label="正向提示词", lines=6, show_copy_button=True)
                    negative_out = gr.Textbox(label="负向提示词", lines=4, show_copy_button=True)
                    params_out = gr.Textbox(label="参数建议", lines=3, show_copy_button=True)
                    pack_out = gr.Textbox(
                        label="复制套装（正向 + 负向 + 参数）",
                        lines=8,
                        show_copy_button=True,
                    )
                    workbench_status = gr.Textbox(
                        label="状态说明（不是历史）",
                        lines=2,
                    )

                    gr.Markdown(
                        "### 历史（本地提示，不是图库）\n"
                        "点「分析并生成」或「按当前场面句生成」后自动记一条。"
                        "「再存一条」只是把当前提示再存一次。"
                    )
                    history_pick = gr.Dropdown(
                        label="最近提示",
                        choices=history_choices("image"),
                        value=None,
                        allow_custom_value=False,
                    )
                    with gr.Row():
                        save_hist_btn = gr.Button("再存一条")
                        load_hist_btn = gr.Button("载入提示")
                        del_hist_btn = gr.Button("删除")

                    gr.Markdown(
                        "### 角色 / 风格档案（可选）\n"
                        "空着不影响分析。要锁同一张脸或同一套画风时再手写，"
                        "或从参考图点「提取草稿」后改掉构图词再保存。"
                    )
                    profile_kind = gr.Radio(
                        label="档案类型",
                        choices=KIND_CHOICES,
                        value="角色",
                    )
                    profile_pick = gr.Dropdown(
                        label="已存档案",
                        choices=profile_choices(),
                        value=None,
                        allow_custom_value=True,
                    )
                    profile_name = gr.Textbox(label="档案名称", placeholder="例如：白发红瞳 / 吉卜力水彩")
                    profile_lock = gr.Textbox(
                        label="锁词（英文短语，不要写构图）",
                        lines=3,
                        placeholder="pale blue hair, red eyes, white outfit",
                    )
                    with gr.Row():
                        extract_btn = gr.Button("从参考图提取草稿")
                        save_profile_btn = gr.Button("保存档案")
                        del_profile_btn = gr.Button("删除档案")

        with gr.Column(visible=False) as video_panel:
            gr.Markdown(
                """
                **视频侧**：一次只写**一个镜头**。参考图请另交给视频模型。
                模板是：在什么环境里，什么人物，在什么音乐下，做什么动作。再套景别、运镜、光线、焦点、画质和时长。
                环境和人物留空时，用标签草稿填上。不写对白，不拆成下一镜。
                """
            )
            with gr.Row():
                with gr.Column(scale=1):
                    video_image = gr.Image(label="画面锁（静帧）", type="pil")
                    video_style = gr.Dropdown(label="写法", choices=STYLE_CHOICES, value="通用")
                    video_shot = gr.Dropdown(label="景别", choices=SHOT_CHOICES, value="跟随场面句")
                    video_camera = gr.Dropdown(label="运镜", choices=CAMERA_CHOICES, value="缓慢推进")
                    video_light = gr.Dropdown(label="光线", choices=LIGHT_CHOICES, value="跟随场面句")
                    video_focus = gr.Dropdown(label="焦点", choices=FOCUS_CHOICES, value="主体清晰")
                    video_motion = gr.Dropdown(label="主体动作", choices=MOTION_CHOICES, value="缓慢")
                    video_grade = gr.Dropdown(label="画质", choices=GRADE_CHOICES, value="清晰稳定")
                    video_ambience = gr.Dropdown(label="环境音", choices=AMBIENCE_CHOICES, value="不写")
                    video_duration = gr.Dropdown(label="时长", choices=DURATION_CHOICES, value="5秒")
                    gr.Markdown("### 这一镜头\n四格套成一句。音乐、动作没有就留空。")
                    video_env = gr.Textbox(label="环境", placeholder="例如：夏日森林")
                    video_subject = gr.Textbox(label="人物 / 物体", placeholder="例如：两个站着的人")
                    video_music = gr.Textbox(label="音乐", placeholder="例如：轻柔钢琴；没有就留空")
                    video_action = gr.Textbox(label="动作", placeholder="例如：缓缓转头看向镜头")
                    video_intent = gr.Textbox(
                        label="气氛备注（可选，不改变镜头和时长）",
                        placeholder="例如：更冷清 / 夜色 / 风吹动头发",
                        lines=3,
                    )
                    video_run = gr.Button("生成视频提示", variant="primary")
                    video_pick = gr.Radio(label="CLIP 场面句候选", choices=[], value=None)
                    video_scene = gr.Textbox(
                        label="场面句（画面锁，可改）",
                        lines=3,
                        placeholder="分析后改成你认可的一句，再生成",
                    )
                    video_apply = gr.Button("按当前场面句生成")
                with gr.Column(scale=1):
                    video_analysis = gr.Textbox(label="视觉分析", lines=8, show_copy_button=True)
                    video_positive = gr.Textbox(label="视频提示", lines=8, show_copy_button=True)
                    video_negative = gr.Textbox(label="负向提示词", lines=4, show_copy_button=True)
                    video_params = gr.Textbox(label="参数", lines=3, show_copy_button=True)
                    video_pack = gr.Textbox(label="复制套装", lines=8, show_copy_button=True)
                    video_status = gr.Textbox(label="状态说明（不是历史）", lines=2)
                    gr.Markdown("### 视频历史\n只记录视频提示，和图片侧历史分开。")
                    video_history = gr.Dropdown(
                        label="最近视频提示",
                        choices=history_choices("video"),
                        value=None,
                        allow_custom_value=False,
                    )
                    with gr.Row():
                        video_save = gr.Button("再存一条")
                        video_load = gr.Button("载入提示")
                        video_del = gr.Button("删除")

        outputs = [
            analysis_out,
            positive_out,
            negative_out,
            params_out,
            pack_out,
            cache,
            scene_pick,
            scene_edit,
            history_pick,
            workbench_status,
        ]
        run_inputs = [
            image,
            character_refs,
            style_refs,
            target_model,
            extra_instruction,
            use_vlm,
            profile_apply,
            cache,
        ]
        run_btn.click(fn=run_pipeline, inputs=run_inputs, outputs=outputs)
        use_vlm.change(fn=on_vlm_toggle, inputs=run_inputs, outputs=outputs)
        scene_pick.change(
            fn=on_pick_candidate,
            inputs=[scene_pick, cache],
            outputs=[scene_edit],
        )
        apply_btn.click(
            fn=apply_scene_sentence,
            inputs=[
                scene_edit,
                character_refs,
                style_refs,
                target_model,
                extra_instruction,
                use_vlm,
                profile_apply,
                cache,
            ],
            outputs=outputs,
        )
        save_hist_btn.click(fn=save_current_history, inputs=[cache], outputs=[history_pick, workbench_status])
        load_hist_btn.click(
            fn=load_history_entry,
            inputs=[history_pick, cache],
            outputs=[*outputs, target_model, extra_instruction],
        )
        del_hist_btn.click(fn=remove_history_entry, inputs=[history_pick], outputs=[history_pick, workbench_status])
        profile_pick.change(
            fn=fill_profile,
            inputs=[profile_pick],
            outputs=[profile_kind, profile_name, profile_lock],
        )
        extract_btn.click(
            fn=extract_lock_draft,
            inputs=[profile_kind, image, character_refs, style_refs, cache],
            outputs=[profile_lock, workbench_status],
        )
        save_profile_btn.click(
            fn=save_profile_entry,
            inputs=[profile_kind, profile_name, profile_lock],
            outputs=[profile_apply, profile_pick, workbench_status],
        )
        del_profile_btn.click(
            fn=remove_profile_entry,
            inputs=[profile_name],
            outputs=[
                profile_apply,
                profile_pick,
                profile_kind,
                profile_name,
                profile_lock,
                workbench_status,
            ],
        )

        video_outputs = [
            video_analysis,
            video_positive,
            video_negative,
            video_params,
            video_pack,
            cache,
            video_pick,
            video_scene,
            video_history,
            video_status,
            video_env,
            video_subject,
            video_music,
            video_action,
        ]
        video_inputs = [
            video_image,
            video_style,
            video_shot,
            video_camera,
            video_light,
            video_focus,
            video_motion,
            video_grade,
            video_ambience,
            video_duration,
            video_env,
            video_subject,
            video_music,
            video_action,
            video_intent,
            cache,
        ]
        side.change(fn=_switch_side, inputs=[side], outputs=[image_panel, video_panel])
        video_run.click(fn=run_video, inputs=video_inputs, outputs=video_outputs)
        video_apply.click(
            fn=apply_video_scene,
            inputs=[
                video_scene,
                video_style,
                video_shot,
                video_camera,
                video_light,
                video_focus,
                video_motion,
                video_grade,
                video_ambience,
                video_duration,
                video_env,
                video_subject,
                video_music,
                video_action,
                video_intent,
                cache,
            ],
            outputs=video_outputs,
        )
        video_pick.change(fn=on_pick_video, inputs=[video_pick, cache], outputs=[video_scene])
        video_save.click(fn=save_video_history, inputs=[cache], outputs=[video_history, video_status])
        video_load.click(
            fn=load_video_history,
            inputs=[video_history, cache],
            outputs=[
                *video_outputs,
                video_style,
                video_shot,
                video_camera,
                video_light,
                video_focus,
                video_motion,
                video_grade,
                video_ambience,
                video_duration,
                video_intent,
            ],
        )
        video_del.click(fn=remove_video_history, inputs=[video_history], outputs=[video_history, video_status])

    return demo


if __name__ == "__main__":
    build_ui().launch()
