from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root() / path


@lru_cache(maxsize=1)
def load_vision_config() -> dict[str, Any]:
    config_path = repo_root() / "configs" / "vision.json"
    with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_tag_vocab() -> dict[str, Any]:
    vocab_path = repo_root() / "configs" / "tag_vocab.json"
    with vocab_path.open(encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_scene_bank() -> dict[str, Any]:
    path = repo_root() / "configs" / "scene_bank.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_llm_config() -> dict[str, Any]:
    config_path = repo_root() / "configs" / "llm.json"
    with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)
