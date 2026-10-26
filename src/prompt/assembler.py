"""按场面句 + 媒介判断组装提示词，参数随构图变化。"""

from __future__ import annotations

from typing import Any

from src.vision.resolve import infer_media_mode, scene_facts, scene_sentence
from src.vision.taxonomy import english_phrases

COMMON_NEGATIVE = (
    "low quality, worst quality, blurry, jpeg artifacts, watermark, signature, "
    "text, logo, cropped, out of frame"
)

STYLE_NEGATIVES = {
    "anime": (
        "photorealistic, photograph, realistic skin, 3d render, cgi, "
        "western cartoon, extra fingers, deformed hands"
    ),
    "photo": (
        "anime, cartoon, illustration, painting, cgi, oversmoothed skin, "
        "plastic look, extra fingers, deformed hands"
    ),
    "paint": (
        "photograph, snapshot, oversharp digital noise, extra fingers, "
        "deformed hands"
    ),
    "graphic": (
        "photorealistic street photo, messy background, extra fingers, "
        "deformed hands, 3d render"
    ),
    "general": (
        "deformed, extra fingers, mutated hands, ugly, poorly drawn face"
    ),
}

TARGET_PARAMS = {
    "SDXL": "分辨率 1024x1024；步数 28；CFG 6.0；采样 DPM++ 2M Karras。",
    "Flux": "分辨率 1024x1024；步数 24；guidance 3.5；适合自然语言提示词。",
    "视频生成（通用）": "分辨率 1280x720；4–6 秒；24fps；中等运动；镜头缓慢横移或推进。",
}

INSTRUCTION_ADD = (
    ("赛博朋克", ("cyberpunk city", "neon lights", "rain-soaked streets")),
    ("cyberpunk", ("cyberpunk city", "neon lights")),
    ("写实", ("photorealistic", "photograph", "natural skin texture")),
    ("更写实", ("photorealistic", "photograph")),
    ("水彩", ("watercolor painting", "soft pigment blooms")),
    ("油画", ("oil painting", "visible brush strokes")),
    ("电影", ("cinematic lighting", "film still")),
    ("暗", ("dark atmosphere", "low-key lighting")),
    ("亮", ("bright lighting", "high-key lighting")),
    ("夜", ("night scene", "moonlight")),
)

INSTRUCTION_REMOVE = {
    "写实": {"anime style", "digital illustration", "comic book style", "anime screencap"},
    "更写实": {"anime style", "digital illustration", "anime screencap"},
}


def _grouped(analysis: dict[str, Any]) -> dict[str, list[tuple[str, float]]]:
    raw = analysis.get("grouped_tags") or {}
    grouped: dict[str, list[tuple[str, float]]] = {}
    for key, items in raw.items():
        cleaned: list[tuple[str, float]] = []
        for item in items:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                cleaned.append((str(item[0]), float(item[1])))
        grouped[str(key)] = cleaned
    return grouped


def _appearance_phrases(analysis: dict[str, Any]) -> list[str]:
    blob = " ".join(
        str(analysis.get(key) or "")
        for key in ("composition", "color", "linework")
    )
    mapping = (
        ("暖色", "warm color palette"),
        ("冷色", "cool color palette"),
        ("高饱和", "vibrant colors"),
        ("低饱和", "muted colors"),
        ("单色", "monochrome"),
        ("偏暗", "low-key lighting"),
        ("偏亮", "high-key lighting"),
        ("高对比", "high contrast"),
        ("低对比", "soft contrast"),
        ("中心", "centered composition"),
        ("三分", "rule of thirds"),
        ("边缘密", "crisp linework"),
        ("边缘少", "soft shapes, painterly edges"),
    )
    return [phrase for key, phrase in mapping if key in blob]


def scene_conflict_bans(sentence: str, mode: str) -> set[str]:
    """场面句已经选定后，丢掉和它打架的标签。"""
    text = " ".join((sentence or "").lower().split())
    bans: set[str] = set()

    has_four = "four " in text
    has_three = "three " in text
    has_two = any(
        token in text
        for token in ("two ", "two people", "two anime", "boy and girl", "girl and boy", "a man and a woman")
    )
    singular = (not has_two and not has_three and not has_four) and any(
        token in text
        for token in (
            "a girl",
            "an anime girl",
            "a woman",
            "a man",
            "a person",
            "an anime character",
            "a studio ghibli girl",
            "a young person",
            "a selfie",
            "a webcam portrait",
        )
    )
    if has_four:
        bans.update({"two people", "three people"})
    elif has_three:
        bans.update({"two people", "four people", "crowd of people"})
    elif has_two:
        bans.update({"three people", "four people", "crowd of people"})
    elif singular:
        bans.update({"two people", "three people", "four people", "crowd of people", "two anime girls"})
        if "child" not in text and "boy" not in text:
            bans.add("child")

    if "mirror" not in text:
        bans.update({"mirror reflection", "looking at herself in a mirror"})
    if "haircut" in text or "barber" in text:
        bans.update({"album cover", "graphic design", "poster"})
    if "album" in text or "record sleeve" in text:
        bans.update({"getting a haircut"})
    if "looking at the camera" in text or "looking at the viewer" in text:
        bans.add("looking away from the camera")
    if "looking away" in text:
        bans.update({"looking at the viewer", "looking at the camera"})
    if "mecha" not in text and "robot" not in text:
        bans.update({"robot or mecha", "mecha"})
    if "telephone" not in text and "phone" not in text:
        bans.add("talking on a telephone")
    if "historical" not in text and mode == "anime":
        bans.add("historical setting")
    if mode == "anime":
        bans.update(
            {
                "film still",
                "photograph",
                "selfie photo",
                "photorealistic image",
                "film grain",
            }
        )
    if mode == "photo":
        bans.update(
            {
                "anime style",
                "anime screencap",
                "digital illustration",
                "comic book style",
                "concept art",
            }
        )
    if mode == "graphic":
        bans.update({"selfie photo", "getting a haircut"})
    return bans


def _blocked(phrase: str, bans: set[str]) -> bool:
    lowered = phrase.lower()
    return any(ban == lowered or ban in lowered or lowered in ban for ban in bans)


def collect_phrases(analysis: dict[str, Any], extra_instruction: str) -> list[str]:
    grouped = _grouped(analysis)
    phrases = english_phrases(grouped)
    phrases.extend(_appearance_phrases(analysis))
    lowered = extra_instruction.lower()
    for key, extras in INSTRUCTION_ADD:
        if key.lower() in extra_instruction or key.lower() in lowered:
            phrases.extend(extras)
    remove: set[str] = set()
    for key, tokens in INSTRUCTION_REMOVE.items():
        if key in extra_instruction:
            remove |= tokens
    unique: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        if phrase in remove or phrase in seen:
            continue
        seen.add(phrase)
        unique.append(phrase)
    return unique


def _mode(analysis: dict[str, Any], extra_instruction: str) -> str:
    mode = str(analysis.get("media_mode") or "")
    if mode in STYLE_NEGATIVES:
        if extra_instruction and any(key in extra_instruction for key in ("写实", "自拍")):
            return "photo"
        return mode
    return infer_media_mode(
        _grouped(analysis),
        analysis.get("appearance") or {},
        extra_instruction,
    )


def suggest_params(analysis: dict[str, Any], target_model: str, mode: str) -> str:
    width = int(analysis.get("image_width") or 1024)
    height = int(analysis.get("image_height") or 1024)
    ratio = width / max(height, 1)
    facts = analysis.get("scene_facts") or scene_facts(_grouped(analysis))
    shot = str(facts.get("shot") or "")

    if target_model.startswith("视频"):
        size = "1280x720" if ratio >= 0.9 else "720x1280"
        return f"分辨率 {size}；4–6 秒；24fps；中等运动。风格判断：{mode}。"

    if mode == "graphic" or "album cover" in str(facts.get("medium") or ""):
        size = "1024x1024"
    elif ratio > 1.25:
        size = "1216x832"
    elif ratio < 0.8:
        size = "832x1216"
    elif "half-body" in shot or "portrait" in shot or "close-up" in shot:
        size = "896x1152"
    else:
        size = "1024x1024"

    if target_model == "Flux":
        return f"分辨率 {size}；步数 24；guidance 3.5。风格判断：{mode}。"
    return f"分辨率 {size}；步数 28；CFG 6.0；采样 DPM++ 2M Karras。风格判断：{mode}。"


def assemble_prompts(
    analysis: dict[str, Any],
    target_model: str,
    extra_instruction: str = "",
) -> dict[str, str]:
    mode = _mode(analysis, extra_instruction)
    facts = analysis.get("scene_facts") or scene_facts(_grouped(analysis))
    sentence = analysis.get("scene_sentence") or scene_sentence(facts, mode)
    phrases = collect_phrases(analysis, extra_instruction)
    from src.llm.ollama_client import usable_vlm_text

    vlm_text = usable_vlm_text(str(analysis.get("vlm_text") or ""))

    banned = scene_conflict_bans(sentence, mode)
    if mode == "photo":
        banned.update({"anime style", "digital illustration", "comic book style", "concept art"})
    if mode == "anime":
        banned.update({"photorealistic image", "photograph", "selfie photo"})
    phrases = [item for item in phrases if item not in banned and not _blocked(item, banned)]

    if target_model == "Flux":
        lead = f"{sentence}. {vlm_text}" if vlm_text else sentence
        positive = lead + ". " + ", ".join(phrases[:8])
        if extra_instruction:
            positive += f". Direction: {extra_instruction}"
    elif target_model.startswith("视频"):
        motion = "gentle ambient motion, slow camera pan"
        if any(token in extra_instruction for token in ("快", "激烈", "dynamic")):
            motion = "dynamic camera movement"
        lead = f"{sentence}. {vlm_text}" if vlm_text else sentence
        positive = f"{lead}, {', '.join(phrases[:6])}"
        if "camera" not in lead.lower() and "pan" not in lead.lower():
            positive += f", {motion}"
        if extra_instruction:
            positive += f". Intent: {extra_instruction}"
    else:
        quality = {
            "anime": ["masterpiece", "best quality", "anime screencap"],
            "photo": ["photorealistic", "sharp focus", "natural lighting"],
            "paint": ["masterpiece", "fine art"],
            "graphic": ["graphic design", "clean layout"],
            "general": ["masterpiece", "best quality"],
        }[mode]
        lead = f"{sentence}. {vlm_text}" if vlm_text else sentence
        body = [lead] + [
            item for item in phrases if item.lower() not in lead.lower()
        ]
        positive = ", ".join(quality + body)
        mapped = any(key in extra_instruction for key, _ in INSTRUCTION_ADD)
        if extra_instruction and extra_instruction not in positive and not mapped:
            positive += f", {extra_instruction}"

    negative = f"{COMMON_NEGATIVE}, {STYLE_NEGATIVES[mode]}"
    if mode == "photo" or any(token in extra_instruction for token in ("写实", "photoreal")):
        negative += ", cartoon, anime, illustration, painting"
    if target_model.startswith("视频"):
        negative += ", flicker, motion blur smear, jitter, still photo freeze"

    params = suggest_params(analysis, target_model, mode)
    if extra_instruction:
        params += f" 已纳入追加指令：{extra_instruction}"

    return {
        "positive": positive,
        "negative": negative,
        "params": params,
    }
