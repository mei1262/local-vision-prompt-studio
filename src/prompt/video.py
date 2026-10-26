"""一条视频提示。画面用已确认的场面句；运镜来自选项，不从静帧推断。"""

from __future__ import annotations

import re
from typing import Any

from src.prompt.assembler import COMMON_NEGATIVE, STYLE_NEGATIVES
from src.vision.resolve import infer_media_mode, scene_facts, scene_sentence

CAMERA_CHOICES = ["固定", "缓慢推进", "缓慢拉远", "横移", "环绕", "轻微手持"]
MOTION_CHOICES = ["几乎静止", "缓慢", "中等", "明显"]
DURATION_CHOICES = ["4秒", "5秒", "6秒", "8秒"]
STYLE_CHOICES = ["通用", "Kling", "Runway", "Luma"]
SHOT_CHOICES = ["跟随场面句", "特写", "近景", "中景", "中全景", "全景", "远景"]
LIGHT_CHOICES = ["跟随场面句", "柔和自然光", "侧光", "逆光", "低调光", "高调光", "黄金时刻"]
FOCUS_CHOICES = ["主体清晰", "全景深", "跟随场面句"]
GRADE_CHOICES = ["清晰稳定", "电影质感", "柔和胶片"]
AMBIENCE_CHOICES = ["不写", "轻环境底噪", "室内安静", "户外自然"]

_CAMERA = {
    "固定": "locked-off camera",
    "缓慢推进": "slow push-in",
    "缓慢拉远": "slow pull-back",
    "横移": "slow lateral move",
    "环绕": "slow arc around the subject",
    "轻微手持": "subtle handheld drift",
}
_MOTION = {
    "几乎静止": "the subject stays nearly still, with only tiny ambient motion",
    "缓慢": "the subject moves slowly",
    "中等": "the subject moves at a moderate pace",
    "明显": "the subject motion is clearly visible",
}
_DURATION = {
    "4秒": "4 seconds",
    "5秒": "5 seconds",
    "6秒": "6 seconds",
    "8秒": "8 seconds",
}
_ARC = {
    "固定": "Start and end on the same framing.",
    "缓慢推进": "Start on this framing and end slightly closer on the same subject.",
    "缓慢拉远": "Start on this framing and end slightly wider on the same subject.",
    "横移": "Start on this framing and end with a small lateral shift, same subject.",
    "环绕": "Start on this framing and end a few degrees around the same subject.",
    "轻微手持": "Keep the same framing, with only a slight handheld drift.",
}
_MARK = {
    "固定": "locked-off",
    "缓慢推进": "push-in",
    "缓慢拉远": "pull-back",
    "横移": "lateral",
    "环绕": "arc",
    "轻微手持": "handheld",
}
_SHOT = {
    "特写": ("close-up", "close-up"),
    "近景": ("medium close-up", "medium close-up"),
    "中景": ("medium shot", "medium shot"),
    "中全景": ("medium-full shot", "medium-full"),
    "全景": ("wide shot", "wide shot"),
    "远景": ("long shot", "long shot"),
}
_LIGHT = {
    "柔和自然光": ("soft natural light", "natural light"),
    "侧光": ("side light", "side light"),
    "逆光": ("backlight", "backlight"),
    "低调光": ("low-key lighting", "low-key"),
    "高调光": ("high-key lighting", "high-key"),
    "黄金时刻": ("golden-hour light", "golden-hour"),
}
_FOCUS = {
    "主体清晰": ("the subject stays sharp, background slightly soft", "subject stays sharp"),
    "全景深": ("deep focus across the frame", "deep focus"),
}
_GRADE = {
    "清晰稳定": ("sharp, stable, and detailed", "sharp, stable"),
    "电影质感": ("cinematic clarity, natural contrast, rich detail", "cinematic clarity"),
    "柔和胶片": ("soft filmic texture, gentle contrast, fine grain", "fine grain"),
}
_AMBIENCE = {
    "轻环境底噪": ("faint ambient room tone, no dialogue", "room tone"),
    "室内安静": ("quiet interior ambience, no dialogue", "quiet interior"),
    "户外自然": ("soft outdoor ambience, no dialogue", "outdoor ambience"),
}

VIDEO_NEGATIVE = (
    "flicker, jitter, morphing, warping, melting, identity change, extra limbs, "
    "sudden cut, jump cut, second shot, scene change, text, subtitles, watermark"
)


def _mode(analysis: dict[str, Any]) -> str:
    mode = str(analysis.get("media_mode") or "")
    if mode in STYLE_NEGATIVES:
        return mode
    grouped = analysis.get("grouped_tags") or {}
    return infer_media_mode(grouped, analysis.get("appearance") or {}, "")


def _sentence(analysis: dict[str, Any], mode: str) -> str:
    facts = analysis.get("scene_facts") or scene_facts(analysis.get("grouped_tags") or {}, mode)
    return str(analysis.get("scene_sentence") or scene_sentence(facts, mode)).strip()


def _pair(table: dict[str, tuple[str, str]], key: str) -> tuple[str, str]:
    return table.get(key, ("", ""))


def _tag_zh(analysis: dict[str, Any], group: str, limit: int = 2) -> str:
    from src.vision.taxonomy import tag_index

    meta = tag_index()
    names: list[str] = []
    for item in (analysis.get("grouped_tags") or {}).get(group) or []:
        if not isinstance(item, (list, tuple)) or not item:
            continue
        info = meta.get(str(item[0]))
        if info and info["zh"] not in names:
            names.append(info["zh"])
        if len(names) >= limit:
            break
    return "、".join(names)


def suggest_slots(analysis: dict[str, Any]) -> dict[str, str]:
    """环境、人物只给草稿。音乐和动作留空，由这一镜头自己填。"""
    return {
        "environment": _tag_zh(analysis, "setting"),
        "subject": _tag_zh(analysis, "subject"),
        "music": "",
        "action": "",
    }


def _clean(text: str) -> str:
    return " ".join((text or "").replace("，", ",").split()).strip(" ,。.")


def _music_phrase(score: str) -> str:
    if any(word in score for word in ("音乐", "配乐", "曲子", "歌曲", "BGM", "bgm")):
        return f"在{score}下"
    return f"在{score}的音乐下"


def _did(actor: str, act: str) -> str:
    if act.startswith(actor):
        return act
    if act.startswith("做了") or act.startswith("做"):
        return f"{actor}{act}"
    return f"{actor}做了{act}"


def shot_narrative(environment: str, subject: str, music: str, action: str) -> str:
    """在什么环境里，什么人物，在什么音乐下，谁做了什么动作。空着的槽不写。"""
    env = _clean(environment)
    who = _clean(subject)
    score = _clean(music)
    act = _clean(action)
    parts: list[str] = []
    if env and who:
        parts.append(f"在{env}里，{who}")
    elif env:
        parts.append(f"在{env}里")
    elif who:
        parts.append(who)
    actor = who or "画面里的主体"
    if score and act:
        parts.append(f"{_music_phrase(score)}，{_did(actor, act)}")
    elif score:
        parts.append(_music_phrase(score))
    elif act and who and parts:
        parts.append(act if act.startswith("做") else f"做了{act}")
    elif act:
        parts.append(_did(actor, act))
    if not parts:
        return ""
    return "，".join(parts) + "。"


_ARC_ZH = {
    "固定": "起幅和落幅保持同一构图",
    "缓慢推进": "起幅是当前构图，落幅略靠近主体",
    "缓慢拉远": "起幅是当前构图，落幅略拉远",
    "横移": "起幅是当前构图，落幅略向一侧移动",
    "环绕": "起幅是当前构图，落幅绕主体转过一个小角度",
    "轻微手持": "构图基本不动，只有轻微手持晃动",
}


def assemble_video(
    analysis: dict[str, Any],
    *,
    style: str = "通用",
    camera: str = "缓慢推进",
    motion: str = "缓慢",
    duration: str = "5秒",
    shot_size: str = "跟随场面句",
    light: str = "跟随场面句",
    focus: str = "主体清晰",
    grade: str = "清晰稳定",
    ambience: str = "不写",
    environment: str = "",
    subject: str = "",
    music: str = "",
    action: str = "",
    extra_instruction: str = "",
) -> dict[str, str]:
    style = style if style in STYLE_CHOICES else "通用"
    camera = camera if camera in _CAMERA else "缓慢推进"
    motion = motion if motion in _MOTION else "缓慢"
    duration = duration if duration in _DURATION else "5秒"
    shot_size = shot_size if shot_size in SHOT_CHOICES else "跟随场面句"
    light = light if light in LIGHT_CHOICES else "跟随场面句"
    focus = focus if focus in FOCUS_CHOICES else "主体清晰"
    grade = grade if grade in GRADE_CHOICES else "清晰稳定"
    ambience = ambience if ambience in AMBIENCE_CHOICES else "不写"
    mode = _mode(analysis)
    sentence = _sentence(analysis, mode)
    narrative = shot_narrative(environment, subject, music, action)
    if not narrative and sentence:
        narrative = f"画面以所附参考图为准。场面参考：{sentence}。"
    elif narrative:
        narrative = f"画面以所附参考图为准。{narrative}"
    else:
        narrative = "画面以所附参考图为准。"
    camera_bits = []
    if shot_size != "跟随场面句":
        camera_bits.append(shot_size)
    camera_bits.append(camera)
    if light != "跟随场面句":
        camera_bits.append(light)
    if focus != "跟随场面句":
        camera_bits.append(focus)
    camera_bits.append(f"主体动作{motion}")
    camera_bits.append(grade)
    if ambience != "不写":
        camera_bits.append(ambience)
    camera_bits.append(duration)
    camera_bits.append("一条镜头，不切到下一镜")
    camera_zh = "，".join(camera_bits) + "。" + _ARC_ZH[camera] + "。"
    camera_en = _CAMERA[camera]
    motion_en = _MOTION[motion]
    duration_en = _DURATION[duration]
    shot_en, shot_mark = _pair(_SHOT, shot_size)
    light_en, light_mark = _pair(_LIGHT, light)
    focus_en, focus_mark = _pair(_FOCUS, focus)
    grade_en, grade_mark = _pair(_GRADE, grade)
    ambience_en, ambience_mark = _pair(_AMBIENCE, ambience)
    mood_text = extra_instruction.strip()
    parts = [narrative, camera_zh]
    if shot_en:
        parts.append(f"Shot size: {shot_en}.")
    parts.append(f"One continuous shot, {duration_en}. Camera: {camera_en}.")
    if light_en:
        parts.append(f"Light: {light_en}.")
    if focus_en:
        parts.append(f"Focus: {focus_en}.")
    parts.append(f"Subject motion: {motion_en}.")
    parts.append(_ARC[camera])
    parts.append("Stay on this shot without a scene change.")
    if grade_en:
        parts.append(f"Image quality: {grade_en}.")
    if ambience_en:
        parts.append(f"Sound: {ambience_en}.")
    if mood_text and not re.search(r"[\u4e00-\u9fff]", mood_text):
        parts.append(f"Mood note: {mood_text}.")
    positive = " ".join(parts)
    if style == "Kling":
        positive += " Stable face and body, natural motion."
    elif style == "Luma":
        positive += " Keep the motion coherent."

    marks = [mark for mark in ( _MARK[camera], duration_en, shot_mark, light_mark, focus_mark, grade_mark, ambience_mark) if mark]
    negative = f"{COMMON_NEGATIVE}, {STYLE_NEGATIVES[mode]}, {VIDEO_NEGATIVE}"
    params = (
        f"写法 {style}；时长 {duration}；景别 {shot_size}；运镜 {camera}；"
        f"光线 {light}；焦点 {focus}；运动 {motion}；画质 {grade}；环境音 {ambience}。"
        f"模板：环境 {environment or '（空）'}；人物 {subject or '（空）'}；"
        f"音乐 {music or '（空）'}；动作 {action or '（空）'}。"
        "参考图另交视频模型。不写对白，不拆分镜。"
    )
    if extra_instruction.strip():
        params += f" 气氛备注：{extra_instruction.strip()}"
    return {
        "positive": positive,
        "negative": negative,
        "params": params,
        "camera_mark": _MARK[camera],
        "duration_en": duration_en,
        "required_marks": marks,
        "scene_sentence": sentence,
        "skip_rewrite": True,
        "environment": environment,
        "subject": subject,
        "music": music,
        "action": action,
    }


def rewrite_keeps_contract(text: str, draft: dict[str, str]) -> bool:
    lowered = " ".join((text or "").lower().split())
    sentence = str(draft.get("scene_sentence") or "").lower()
    words = [word for word in sentence.split() if len(word) > 3][:6]
    if words and not all(word in lowered for word in words):
        return False
    for mark in draft.get("required_marks") or []:
        if str(mark).lower() not in lowered:
            return False
    if any(token in lowered for token in ("storyboard", "shot 1", "shot 2", "cut to", "then cut")):
        return False
    return True
