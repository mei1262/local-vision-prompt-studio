"""历史与档案：只存提示和锁词，不把原图当作再生成素材。"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.vision.paths import repo_root

MAX_HISTORY = 80


def workbench_dir() -> Path:
    path = repo_root() / "outputs" / "workbench"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _read_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def history_path() -> Path:
    return workbench_dir() / "history.jsonl"


def profiles_path() -> Path:
    return workbench_dir() / "profiles.json"


def load_history() -> list[dict[str, Any]]:
    path = history_path()
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("id"):
            items.append(item)
    return items[-MAX_HISTORY:]


def save_history(items: list[dict[str, Any]]) -> None:
    clipped = items[-MAX_HISTORY:]
    text = "\n".join(json.dumps(item, ensure_ascii=False) for item in clipped)
    history_path().write_text(text + ("\n" if clipped else ""), encoding="utf-8")


def history_label(item: dict[str, Any]) -> str:
    stamp = str(item.get("created_at") or "")[:16].replace("T", " ")
    title = str(item.get("title") or "未命名")[:42]
    target = str(item.get("target_model") or "")
    prefix = target.split("（")[0] if target else ""
    side_name = "视频" if item.get("side") == "video" else "图片"
    return f"{side_name} · {stamp} · {prefix} · {title}".strip(" ·")


def history_choices(side: str | None = None) -> list[str]:
    items = load_history()
    if side:
        items = [item for item in items if (item.get("side") or "image") == side]
    return [history_label(item) for item in reversed(items)]


def find_history(label: str) -> dict[str, Any] | None:
    for item in load_history():
        if history_label(item) == label:
            return item
    return None


def append_history(entry: dict[str, Any]) -> list[str]:
    items = load_history()
    last = items[-1] if items else None
    side = entry.get("side") or "image"
    if (
        last
        and (last.get("side") or "image") == side
        and last.get("positive") == entry.get("positive")
        and last.get("scene_sentence") == entry.get("scene_sentence")
        and last.get("target_model") == entry.get("target_model")
    ):
        return history_choices(side)
    title = str(entry.get("scene_sentence") or entry.get("positive") or "提示").strip()
    items.append(
        {
            "id": uuid.uuid4().hex,
            "created_at": _now(),
            "side": side,
            "title": title[:80],
            "target_model": entry.get("target_model") or "",
            "scene_sentence": entry.get("scene_sentence") or "",
            "extra_instruction": entry.get("extra_instruction") or "",
            "positive": entry.get("positive") or "",
            "negative": entry.get("negative") or "",
            "params": entry.get("params") or "",
            "profile_names": list(entry.get("profile_names") or []),
            "lock_notes": entry.get("lock_notes") or "",
            "camera": entry.get("camera") or "",
            "motion": entry.get("motion") or "",
            "duration": entry.get("duration") or "",
            "shot_size": entry.get("shot_size") or "",
            "light": entry.get("light") or "",
            "focus": entry.get("focus") or "",
            "grade": entry.get("grade") or "",
            "ambience": entry.get("ambience") or "",
            "environment": entry.get("environment") or "",
            "subject": entry.get("subject") or "",
            "music": entry.get("music") or "",
            "action": entry.get("action") or "",
        }
    )
    save_history(items)
    return history_choices(side)


def delete_history(label: str, side: str = "image") -> list[str]:
    keep = [item for item in load_history() if history_label(item) != label]
    save_history(keep)
    return history_choices(side)


def load_profiles() -> list[dict[str, Any]]:
    raw = _read_json(profiles_path(), {"profiles": []})
    items = raw.get("profiles") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict) and item.get("name")]


def save_profiles(items: list[dict[str, Any]]) -> None:
    _write_json(profiles_path(), {"profiles": items})


def profile_choices() -> list[str]:
    return [str(item["name"]) for item in load_profiles()]


def find_profile(name: str) -> dict[str, Any] | None:
    for item in load_profiles():
        if item.get("name") == name:
            return item
    return None


def upsert_profile(kind: str, name: str, lock_text: str, notes: str = "") -> list[str]:
    clean_name = (name or "").strip()
    clean_lock = (lock_text or "").strip()
    if not clean_name or not clean_lock:
        return profile_choices()
    kind = "style" if kind == "style" else "character"
    items = load_profiles()
    updated = False
    for item in items:
        if item.get("name") == clean_name:
            item["kind"] = kind
            item["lock_text"] = clean_lock
            item["notes"] = notes
            item["updated_at"] = _now()
            updated = True
            break
    if not updated:
        items.append(
            {
                "id": uuid.uuid4().hex,
                "kind": kind,
                "name": clean_name,
                "lock_text": clean_lock,
                "notes": notes,
                "created_at": _now(),
            }
        )
    save_profiles(items)
    return profile_choices()


def delete_profile(name: str) -> list[str]:
    save_profiles([item for item in load_profiles() if item.get("name") != name])
    return profile_choices()


def selected_lock_texts(names: list[str] | None) -> list[str]:
    wanted = set(names or [])
    texts: list[str] = []
    for item in load_profiles():
        if item.get("name") in wanted and item.get("lock_text"):
            texts.append(str(item["lock_text"]))
    return texts
