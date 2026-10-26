"""本地 Ollama 客户端，只访问 127.0.0.1，不走云端。"""

from __future__ import annotations

import json
from typing import Any

import requests

from src.vision.paths import load_llm_config


def _host() -> str:
    return str(load_llm_config().get("host") or "http://127.0.0.1:11434").rstrip("/")


def is_available() -> bool:
    try:
        response = requests.get(f"{_host()}/api/tags", timeout=2)
        return response.ok
    except requests.RequestException:
        return False


def list_models() -> list[str]:
    try:
        response = requests.get(f"{_host()}/api/tags", timeout=5)
        response.raise_for_status()
        items = response.json().get("models") or []
        return [str(item.get("name") or "") for item in items if item.get("name")]
    except requests.RequestException:
        return []


def pick_vlm(explicit: str | None = None) -> str | None:
    installed = list_models()
    if not installed:
        return None
    names = [explicit] if explicit else list(load_llm_config().get("preferred_vlm") or [])
    for candidate in names:
        if not candidate:
            continue
        for name in installed:
            if name == candidate or name.startswith(candidate):
                return name
    return None


def pick_model(explicit: str | None = None) -> str | None:
    installed = list_models()
    if not installed:
        return None
    if explicit:
        for name in installed:
            if name == explicit or name.startswith(explicit):
                return name
        return None
    for candidate in list(load_llm_config().get("preferred_models") or []):
        for name in installed:
            if name == candidate or name.startswith(candidate):
                return name
    return installed[0]


def generate_text(prompt: str, model: str | None = None) -> str:
    chosen = pick_model(model)
    if not chosen:
        return "（待接入 Ollama）本机没有可用的本地语言模型。"

    config = load_llm_config()
    payload = {
        "model": chosen,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(config.get("temperature", 0.5)),
            "num_ctx": int(config.get("num_ctx", 2048)),
        },
    }
    try:
        response = requests.post(
            f"{_host()}/api/generate",
            json=payload,
            timeout=int(config.get("timeout_sec", 90)),
        )
        response.raise_for_status()
        return str(response.json().get("response") or "").strip()
    except requests.RequestException as exc:
        return f"（Ollama 调用失败）{exc}"


def generate_json(prompt: str, model: str | None = None) -> dict[str, Any] | None:
    chosen = pick_model(model)
    if not chosen:
        return None

    config = load_llm_config()
    payload = {
        "model": chosen,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": float(config.get("temperature", 0.5)),
            "num_ctx": int(config.get("num_ctx", 2048)),
        },
    }
    try:
        response = requests.post(
            f"{_host()}/api/generate",
            json=payload,
            timeout=int(config.get("timeout_sec", 90)),
        )
        response.raise_for_status()
        raw = str(response.json().get("response") or "").strip()
        return _parse_json_object(raw)
    except (requests.RequestException, ValueError, json.JSONDecodeError):
        return None


def usable_vlm_text(text: str) -> str:
    """丢掉过短、乱码、复读、纯数字的看图结果。"""
    import re

    cleaned = " ".join((text or "").replace("\x00", " ").split())
    if len(cleaned) < 32:
        return ""
    cjk = sum(1 for ch in cleaned if "\u4e00" <= ch <= "\u9fff")
    if cjk >= 6:
        return ""
    letters = re.findall(r"[A-Za-z]{2,}", cleaned)
    if len(letters) < 8:
        return ""
    if len(set(w.lower() for w in letters)) < 6:
        return ""
    if sum(ch.isdigit() for ch in cleaned) / len(cleaned) > 0.12:
        return ""
    lowered = cleaned.lower()
    if lowered.startswith("ids") or " ids." in lowered:
        return ""
    anchors = (
        "girl", "woman", "man", "person", "people", "anime", "photo",
        "hair", "wearing", "holding", "looking", "sitting", "standing",
        "room", "phone", "camera", "portrait",
    )
    if not any(word in lowered for word in anchors):
        return ""
    if len(cleaned) > 400:
        cleaned = cleaned[:400].rsplit(" ", 1)[0]
    return cleaned


def describe_image(
    image_jpeg: bytes,
    prompt: str,
    model: str | None = None,
) -> str:
    """用本机视觉模型看图。image_jpeg 为 JPEG 字节。失败或乱码时返回空串。"""
    import base64

    chosen = pick_vlm(model)
    if not chosen:
        return ""
    config = load_llm_config()
    encoded = base64.b64encode(image_jpeg).decode("ascii")
    options = {
        "temperature": 0.1,
        "num_ctx": 1024,
        "num_predict": 90,
    }
    timeout = int(config.get("vlm_timeout_sec", 180))
    try:
        response = requests.post(
            f"{_host()}/api/generate",
            json={
                "model": chosen,
                "prompt": prompt,
                "images": [encoded],
                "stream": False,
                "options": options,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        raw = str(response.json().get("response") or "").strip()
        return usable_vlm_text(raw)
    except requests.RequestException:
        return ""


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("响应中没有 JSON 对象")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("JSON 根节点不是对象")
    return data
