from __future__ import annotations

from typing import Any

from .paths import load_tag_vocab, load_vision_config


def flatten_tags() -> list[dict[str, str]]:
    vocab = load_tag_vocab()
    tags: list[dict[str, str]] = []
    for group_name, items in vocab["groups"].items():
        for item in items:
            tags.append(
                {
                    "id": item["id"],
                    "en": item["en"],
                    "zh": item["zh"],
                    "group": group_name,
                }
            )
    return tags


def tag_index() -> dict[str, dict[str, str]]:
    return {item["id"]: item for item in flatten_tags()}


def ranked_pairs(
    tag_ids: list[str],
    scores: list[float],
) -> list[tuple[str, float]]:
    meta = tag_index()
    pairs = [
        (tag_id, float(score))
        for tag_id, score in zip(tag_ids, scores)
        if tag_id in meta
    ]
    pairs.sort(key=lambda item: item[1], reverse=True)
    return pairs


def score_tags(
    tag_ids: list[str],
    scores: list[float],
) -> dict[str, list[tuple[str, float]]]:
    """按相对排名选标签，避免绝对阈值把所有词滤光。"""
    config = load_vision_config()
    threshold = float(config.get("score_threshold", 0.10))
    relative = float(config.get("relative_floor", 0.86))
    always_top_k = int(config.get("always_top_k", 10))
    per_group = int(config.get("max_tags_per_group", 3))
    meta = tag_index()
    pairs = ranked_pairs(tag_ids, scores)
    if not pairs:
        return {}

    best = pairs[0][1]
    floor = max(threshold, best * relative)
    kept: set[str] = {tag_id for tag_id, _score in pairs[:always_top_k]}
    for tag_id, score in pairs:
        if score >= floor:
            kept.add(tag_id)

    grouped: dict[str, list[tuple[str, float]]] = {}
    for tag_id, score in pairs:
        if tag_id not in kept:
            continue
        grouped.setdefault(meta[tag_id]["group"], []).append((tag_id, score))

    for group_name, items in grouped.items():
        grouped[group_name] = items[:per_group]
    return grouped


def format_top_candidates(tag_ids: list[str], scores: list[float], limit: int = 8) -> str:
    meta = tag_index()
    parts = []
    for tag_id, score in ranked_pairs(tag_ids, scores)[:limit]:
        info = meta[tag_id]
        parts.append(f"{info['zh']} {score:.3f}")
    return "、".join(parts)


def format_group(grouped: dict[str, list[tuple[str, float]]], group_name: str) -> str:
    meta = tag_index()
    items = grouped.get(group_name) or []
    if not items:
        return ""
    parts = []
    for tag_id, score in items:
        info = meta[tag_id]
        parts.append(f"{info['zh']}({info['en']}, {score:.2f})")
    return "、".join(parts)


def style_tag_labels(grouped: dict[str, list[tuple[str, float]]]) -> list[str]:
    config = load_vision_config()
    limit = int(config.get("max_style_tags", 12))
    meta = tag_index()
    picked: list[tuple[str, float]] = []
    for group_name in ("style", "medium", "lighting", "mood"):
        picked.extend(grouped.get(group_name) or [])
    picked.sort(key=lambda pair: pair[1], reverse=True)
    labels: list[str] = []
    for tag_id, _score in picked[:limit]:
        labels.append(meta[tag_id]["zh"])
    return labels


def english_phrases(
    grouped: dict[str, list[tuple[str, float]]],
    groups: tuple[str, ...] | None = None,
) -> list[str]:
    meta = tag_index()
    order = groups or (
        "subject",
        "setting",
        "shot",
        "composition",
        "style",
        "medium",
        "lighting",
        "color",
        "action",
        "mood",
        "quality",
    )
    phrases: list[str] = []
    seen: set[str] = set()
    for group_name in order:
        for tag_id, _score in grouped.get(group_name) or []:
            info = meta.get(tag_id)
            if not info:
                continue
            text = info["en"]
            for prefix in ("a photo of ", "an illustration of ", "a picture of ", "a ", "an ", "the "):
                if text.startswith(prefix):
                    text = text[len(prefix) :]
                    break
            if text and text not in seen:
                seen.add(text)
                phrases.append(text)
    return phrases


def build_content_text(grouped: dict[str, list[tuple[str, float]]]) -> str:
    chunks = []
    for group_name, title in (
        ("subject", "主体"),
        ("setting", "场景"),
        ("shot", "景别"),
        ("action", "动作"),
        ("mood", "情绪"),
    ):
        text = format_group(grouped, group_name)
        if text:
            chunks.append(f"{title}：{text}")
    return "；".join(chunks) if chunks else "模型未给出足够高置信度的内容标签。"
