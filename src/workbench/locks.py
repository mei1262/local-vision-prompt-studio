"""从参考图抽出身份/风格锁。构图仍以本镜为准，用来出新图而不是复刻原图。"""

from __future__ import annotations

from typing import Any

from src.vision.taxonomy import english_phrases

SKIP_SUBJECT = {
    "landscape",
    "cityscape",
    "interior",
    "architecture",
    "vehicle",
    "food",
    "still_life",
    "crowd",
}

CHARACTER_GROUPS = ("subject", "color")
STYLE_GROUPS = ("style", "medium", "lighting", "mood", "color")


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


def _split_lock(text: str) -> list[str]:
    parts: list[str] = []
    for chunk in (text or "").replace("；", ",").replace(";", ",").split(","):
        piece = chunk.strip(" .")
        if piece and piece not in parts:
            parts.append(piece)
    return parts


def character_lock_from_analysis(analysis: dict[str, Any]) -> str:
    grouped = _grouped(analysis)
    filtered = dict(grouped)
    filtered["subject"] = [
        pair for pair in grouped.get("subject") or [] if pair[0] not in SKIP_SUBJECT
    ]
    phrases = english_phrases(filtered, CHARACTER_GROUPS)
    return ", ".join(phrases[:10])


def style_lock_from_analysis(analysis: dict[str, Any]) -> str:
    phrases = english_phrases(_grouped(analysis), STYLE_GROUPS)
    return ", ".join(phrases[:12])


def merge_locks(
    analysis: dict[str, Any],
    *,
    character_analyses: list[dict[str, Any]] | None = None,
    style_analyses: list[dict[str, Any]] | None = None,
    profile_texts: list[str] | None = None,
) -> dict[str, Any]:
    """保留本镜场面句，只把外貌/画风短语叠进提示。"""
    merged = dict(analysis)
    phrases: list[str] = []
    notes: list[str] = []

    for item in character_analyses or []:
        text = character_lock_from_analysis(item)
        if text:
            phrases.extend(_split_lock(text))
            notes.append(f"角色参考锁：{text}")

    for item in style_analyses or []:
        text = style_lock_from_analysis(item)
        if text:
            phrases.extend(_split_lock(text))
            notes.append(f"风格参考锁：{text}")

    for text in profile_texts or []:
        bits = _split_lock(text)
        if bits:
            phrases.extend(bits)
            notes.append(f"档案锁：{', '.join(bits[:8])}")

    unique: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(phrase)

    merged["lock_phrases"] = unique
    merged["lock_notes"] = " ".join(notes)
    return merged
