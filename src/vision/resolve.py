"""冲突标签清理，以及用规则约束「照片 / 动漫」判断。"""

from __future__ import annotations

from typing import Any

from .taxonomy import tag_index

CONFLICT_SETS = (
    frozenset({"close_up", "portrait_shot", "half_body", "full_body", "wide_shot", "aerial"}),
    frozenset({"looking_at_viewer", "looking_away", "back_view", "side_view", "from_above", "from_below"}),
    frozenset(
        {
            "photorealistic",
            "anime",
            "oil_painting",
            "watercolor",
            "sketch",
            "3d_render",
            "pixel_art",
            "comic",
            "impressionist",
        }
    ),
    frozenset(
        {
            "photograph",
            "digital_painting",
            "traditional_painting",
            "film_still",
            "poster",
            "album_cover",
            "selfie",
        }
    ),
    frozenset({"centered", "rule_of_thirds"}),
    frozenset({"two_people", "three_people", "four_people", "crowd"}),
)

PHOTO_IDS = frozenset({"photorealistic", "photograph", "selfie", "film_still"})
ANIME_IDS = frozenset({"anime", "illustration", "comic", "concept_art"})
PAINT_IDS = frozenset({"oil_painting", "watercolor", "ink_wash", "impressionist", "sketch"})
GRAPHIC_IDS = frozenset({"poster", "album_cover", "graphic_design"})
PERSON_IDS = frozenset({"person", "woman", "man", "child", "portrait_subject"})


def _flat_scores(grouped: dict[str, list[tuple[str, float]]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for items in grouped.values():
        for tag_id, score in items:
            scores[tag_id] = max(score, scores.get(tag_id, 0.0))
    return scores


def _best(scores: dict[str, float], ids: frozenset[str]) -> float:
    hits = [scores[tag_id] for tag_id in ids if tag_id in scores]
    return max(hits) if hits else 0.0


def resolve_conflicts(
    grouped: dict[str, list[tuple[str, float]]],
) -> dict[str, list[tuple[str, float]]]:
    scores = _flat_scores(grouped)
    drop: set[str] = set()
    for group in CONFLICT_SETS:
        present = [(tag_id, scores[tag_id]) for tag_id in group if tag_id in scores]
        if len(present) <= 1:
            continue
        present.sort(key=lambda item: item[1], reverse=True)
        winner = present[0][0]
        for tag_id, _score in present[1:]:
            drop.add(tag_id)
        _ = winner

    person = _best(scores, PERSON_IDS)
    if scores.get("mecha", 0) and person > 0 and scores["mecha"] < person + 0.04:
        drop.add("mecha")

    cleaned: dict[str, list[tuple[str, float]]] = {}
    for group_name, items in grouped.items():
        kept = [(tag_id, score) for tag_id, score in items if tag_id not in drop]
        if kept:
            cleaned[group_name] = kept
    return cleaned


def infer_media_mode(
    grouped: dict[str, list[tuple[str, float]]],
    appearance: dict[str, Any],
    extra_instruction: str = "",
) -> str:
    if any(key in extra_instruction for key in ("写实", "自拍", "照片")):
        return "photo"
    if any(key in extra_instruction for key in ("动漫", "插画", "二次元")):
        return "anime"

    scores = _flat_scores(grouped)
    clear_photo = _best(scores, frozenset({"photograph", "selfie"}))
    filmish = _best(scores, frozenset({"photorealistic", "film_still"}))
    photo = max(clear_photo, filmish)
    anime = _best(scores, ANIME_IDS)
    paint = _best(scores, PAINT_IDS)
    graphic = _best(scores, GRAPHIC_IDS)
    saturation = float(appearance.get("saturation") or 0)
    contrast = float(appearance.get("contrast") or 0)
    edge = float(appearance.get("edge_mean") or 0)

    selfie_like = (
        "selfie" in scores
        or (
            _best(scores, PERSON_IDS) > 0
            and "looking_at_viewer" in scores
            and clear_photo >= anime - 0.01
            and graphic < 0.17
            and contrast >= 0.12
            and edge >= 0.12
        )
    )
    if selfie_like:
        return "photo"

    if graphic >= 0.11 and saturation >= 0.25 and any(tag_id in scores for tag_id in GRAPHIC_IDS):
        return "graphic"

    if anime >= 0.12 and clear_photo < anime + 0.02:
        return "anime"

    if paint >= 0.12 and paint >= max(clear_photo, anime):
        return "paint"

    if clear_photo >= 0.10 or (photo >= 0.10 and anime < 0.12):
        return "photo"

    if graphic >= 0.14 and saturation >= 0.20:
        return "graphic"
    return "general"


def scene_facts(grouped: dict[str, list[tuple[str, float]]], mode: str = "") -> dict[str, str]:
    meta = tag_index()
    scores = _flat_scores(grouped)

    def first(group_name: str) -> str:
        items = grouped.get(group_name) or []
        if not items:
            return ""
        info = meta.get(items[0][0])
        return info["en"] if info else ""

    count = ""
    top_subject = (grouped.get("subject") or [(None, 0.0)])[0][0]
    if top_subject in {"four_people", "three_people", "two_people", "crowd"}:
        count = meta[top_subject]["en"]
    facing = ""
    for tag_id in ("back_view", "looking_away", "side_view", "looking_at_viewer"):
        if tag_id in scores:
            facing = meta[tag_id]["en"]
            break
    medium = first("medium")
    if mode == "photo" and "selfie" in scores:
        medium = meta["selfie"]["en"]
    elif mode == "photo" and any(tag_id in scores for tag_id in GRAPHIC_IDS):
        medium = "a photograph"
    elif mode == "graphic" and "album_cover" in scores:
        medium = meta["album_cover"]["en"]
    if mode == "anime" and medium in {"a photograph", "a film still", "a selfie photo"}:
        medium = "an anime illustration"
    setting = "" if mode == "graphic" else first("setting")
    return {
        "subject": first("subject") or count,
        "count": count,
        "facing": facing,
        "setting": setting,
        "shot": first("shot"),
        "style": first("style"),
        "medium": medium,
    }


_COUNT_PHRASES = (
    ("four people", 4),
    ("four men", 4),
    ("four women", 4),
    ("three people", 3),
    ("three men", 3),
    ("two women", 2),
    ("two men", 2),
    ("two people", 2),
    ("a man and a woman", 2),
    ("a woman and a man", 2),
    ("couple", 2),
)

_CLIP_COUNT = (("four_people", 4), ("three_people", 3), ("two_people", 2), ("crowd", 6))
_INVENTED_ANIMALS = (("cat", "cat"), ("dog", "dog"), ("bird", "bird"))


def _caption_people_count(text: str) -> int | None:
    lowered = text.lower()
    for phrase, count in _COUNT_PHRASES:
        if phrase in lowered:
            return count
    if "crowd" in lowered:
        return 6
    if any(token in lowered for token in ("a woman", "a man", "a person", "a girl", "a boy")):
        return 1
    return None


def judge_caption(
    caption: str,
    grouped: dict[str, list[tuple[str, float]]],
    mode: str,
) -> tuple[bool, str]:
    """轻量描述经常编造，只有和 CLIP 不打架时才采用。"""
    text = caption.strip()
    if len(text) < 8:
        return False, "描述太短，未采用"
    scores = _flat_scores(grouped)
    lowered = text.lower()

    for word, tag_id in _INVENTED_ANIMALS:
        if word in lowered and scores.get(tag_id, 0) < 0.12:
            return False, f"描述编造了{word}，未采用"

    caption_count = _caption_people_count(text)
    clip_count = next((count for tag_id, count in _CLIP_COUNT if tag_id in scores), None)
    if caption_count and clip_count and caption_count != clip_count:
        return False, "描述人数和标签不一致，未采用"
    if "mirror" in scores and caption_count and caption_count >= 2:
        return False, "把镜子倒影说成两个人，未采用"

    if mode == "anime" and any(
        token in lowered for token in ("black and white photo", "black and white", "photograph")
    ):
        return False, "把插画说成照片，未采用"
    if mode in {"photo", "graphic"} and any(token in lowered for token in ("anime", "cartoon")):
        return False, "把照片说成动漫，未采用"
    if "collage" in lowered and "album_cover" in scores:
        return False, "把封面说成拼贴，未采用"
    if any(token in lowered for token in ("pulled", "pulling", "being pulled")):
        return False, "描述动作不可信，未采用"
    return True, "描述与标签大致一致，已采用"


def pick_clip_scene(image_vec, backend) -> dict[str, Any]:
    """用同一套 CLIP 在场面句库里检索，不是另一个描述模型。"""
    vectors = getattr(backend, "scene_vectors", None)
    if vectors is None or not getattr(backend, "scene_ids", None):
        return {}
    scores = (image_vec @ vectors.T).reshape(-1)
    order = scores.argsort()[::-1]
    best = int(order[0])
    second = float(scores[order[1]]) if len(order) > 1 else 0.0
    return {
        "id": backend.scene_ids[best],
        "text": backend.scene_texts[best],
        "mode": backend.scene_modes[best],
        "score": float(scores[best]),
        "margin": float(scores[best]) - second,
        "candidates": [
            {
                "id": backend.scene_ids[int(idx)],
                "text": backend.scene_texts[int(idx)],
                "mode": backend.scene_modes[int(idx)],
                "score": float(scores[int(idx)]),
            }
            for idx in order[:3]
        ],
    }


def scene_sentence(facts: dict[str, str], mode: str) -> str:
    who = facts.get("count") or facts.get("subject") or "a subject"
    facing = facts.get("facing")
    setting = facts.get("setting")
    shot = facts.get("shot")
    medium = facts.get("medium") or {
        "photo": "a photograph",
        "anime": "an anime illustration",
        "paint": "a painting",
        "graphic": "a graphic design image",
        "general": "an image",
    }.get(mode, "an image")
    parts = [medium]
    if shot:
        parts.append(shot)
    parts.append(f"of {who}")
    if facing:
        parts.append(facing)
    if setting:
        parts.append(f"in {setting}")
    return ", ".join(parts)
