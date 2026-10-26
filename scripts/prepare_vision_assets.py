"""下载 CLIP ONNX（ViT-L/14 与回退用 ViT-B/16）并预计算标签向量。"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.embeddings import bake_scene_embeddings, bake_tag_embeddings
from src.vision.paths import load_vision_config, resolve_path
from src.vision.runtime import available_providers, create_session, probe_image_dim

OFFICIAL_ENDPOINT = "https://huggingface.co"
MIRROR_ENDPOINT = "https://hf-mirror.com"


def resolve_endpoints(choice: str) -> list[str]:
    env_endpoint = (os.environ.get("HF_ENDPOINT") or "").rstrip("/")
    if choice == "official":
        return [OFFICIAL_ENDPOINT]
    if choice == "mirror":
        return [MIRROR_ENDPOINT]
    endpoints: list[str] = []
    if env_endpoint:
        endpoints.append(env_endpoint)
    # 国内直连 huggingface.co 常被拒绝，自动优先镜像
    endpoints.extend([MIRROR_ENDPOINT, OFFICIAL_ENDPOINT])
    seen: set[str] = set()
    unique: list[str] = []
    for item in endpoints:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def disable_proxies() -> None:
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def download_file(url: str, dest: Path, retries: int = 5) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"已存在，跳过：{dest}")
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        headers = {}
        done = tmp.stat().st_size if tmp.is_file() else 0
        if done:
            headers["Range"] = f"bytes={done}-"
        print(f"下载 {url}" + (f" （续传 {done / 1e6:.1f} MB，第 {attempt} 次）" if done or attempt > 1 else ""))
        try:
            with requests.get(url, stream=True, timeout=60, headers=headers) as response:
                if response.status_code == 416:
                    tmp.replace(dest)
                    return dest
                response.raise_for_status()
                total = int(response.headers.get("content-length") or 0) + (
                    done if response.status_code == 206 else 0
                )
                mode = "ab" if response.status_code == 206 and done else "wb"
                if mode == "wb":
                    done = 0
                with tmp.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        done += len(chunk)
                        if total:
                            percent = min(100.0, done * 100 / total)
                            print(f"\r  {percent:5.1f}%  {done / 1e6:.1f}/{total / 1e6:.1f} MB", end="")
            print()
            tmp.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"\n  失败：{exc}")
            time.sleep(min(8, attempt * 2))

    raise RuntimeError(f"下载失败：{url}") from last_error


def download_repo_file(repo: str, filename: str, dest: Path, endpoints: list[str]) -> Path:
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"已存在，跳过：{dest}")
        return dest

    errors: list[str] = []
    for endpoint in endpoints:
        url = f"{endpoint}/{repo}/resolve/main/{filename}"
        try:
            return download_file(url, dest)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{endpoint}: {exc}")
            print(f"切换端点重试（当前失败：{endpoint}）")

    raise RuntimeError(
        "无法下载模型文件。请检查网络，或使用镜像：\n"
        "  python scripts/prepare_vision_assets.py --source mirror\n"
        "若本机开了失效代理，可加 --no-proxy。\n"
        + "\n".join(errors)
    )


def pick_provider() -> str:
    providers = available_providers()
    for name in ("DmlExecutionProvider", "CPUExecutionProvider"):
        if name in providers:
            print(f"准备脚本使用执行器：{name}")
            return name
    raise RuntimeError(f"没有可用的 ONNX 执行器：{providers}")


def bake(model: dict, onnx_path: Path, provider: str, image_size: int) -> Path:
    tokenizer_path = resolve_path(model["tokenizer"])
    embed_path = resolve_path(model["embeddings"])
    session = create_session(onnx_path, provider)
    try:
        bake_tag_embeddings(session, tokenizer_path, embed_path, image_size=image_size)
        if model.get("scene_embeddings"):
            bake_scene_embeddings(
                session,
                tokenizer_path,
                resolve_path(model["scene_embeddings"]),
                image_size=image_size,
            )
            print(f"已写入场面向量：{model['scene_embeddings']}")
    finally:
        del session
    print(f"已写入标签向量：{embed_path}")
    return embed_path


def vision_matches_embeddings(model: dict, provider: str, image_size: int) -> bool:
    import numpy as np

    vision_path = resolve_path(model["vision_onnx"])
    embed_path = resolve_path(model["embeddings"])
    data = np.load(embed_path, allow_pickle=True)
    tag_dim = int(data["vectors"].shape[-1])
    session = create_session(vision_path, provider)
    try:
        dim = probe_image_dim(session, image_size)
    finally:
        del session
    print(f"{model['label']} vision 维度 {dim}，标签向量维度 {tag_dim}")
    return dim == tag_dim


def prepare_model(
    model: dict,
    provider: str,
    image_size: int,
    force_bake: bool,
    endpoints: list[str],
) -> None:
    vision_path = resolve_path(model["vision_onnx"])
    text_path = resolve_path(model["text_onnx"])
    joint_path = resolve_path(model["joint_onnx"])
    embed_path = resolve_path(model["embeddings"])
    tokenizer_path = resolve_path(model["tokenizer"])
    repo = model["repo"]

    download_repo_file(repo, model["remote_vision"], vision_path, endpoints)
    download_repo_file(repo, model["remote_text"], text_path, endpoints)
    download_repo_file(repo, model["remote_tokenizer"], tokenizer_path, endpoints)

    if force_bake or not embed_path.is_file():
        print(f"用文本塔预计算 {model['label']} 标签向量…")
        bake(model, text_path, provider, image_size)
    else:
        print(f"{model['label']} 标签向量已存在：{embed_path}")

    if vision_matches_embeddings(model, provider, image_size):
        return

    print(f"{model['label']} 视觉塔与文本向量不在同一空间，改用量化联合模型。")
    download_repo_file(repo, model["remote_joint"], joint_path, endpoints)
    print(f"用联合模型重新计算 {model['label']} 标签向量…")
    bake(model, joint_path, provider, image_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="下载视觉 ONNX 并生成标签向量。")
    parser.add_argument("--force-bake", action="store_true", help="即使已有 npz 也重新计算标签向量")
    parser.add_argument(
        "--purge-text",
        action="store_true",
        help="生成向量后删除文本 ONNX，大约再省 0.4–0.5GB",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "mirror", "official"),
        default="auto",
        help="auto 先走 hf-mirror.com，失败再试官网；国内请用 mirror",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="忽略 HTTP(S)_PROXY。本机代理没开却仍被设置时，会出现连接被拒绝",
    )
    args = parser.parse_args()

    if args.no_proxy:
        disable_proxies()
        print("已忽略代理环境变量。")

    endpoints = resolve_endpoints(args.source)
    print("下载端点：" + " → ".join(endpoints))

    config = load_vision_config()
    provider = pick_provider()
    image_size = int(config.get("image_size", 224))

    for model in config["models"]:
        print(f"\n=== {model['label']} ===")
        prepare_model(model, provider, image_size, args.force_bake, endpoints)
        if args.purge_text:
            text_path = resolve_path(model["text_onnx"])
            if text_path.is_file():
                text_path.unlink()
                print(f"已删除文本模型：{text_path}")

    print("\n完成。之后运行 python app.py 即可做本地视觉分析。")


if __name__ == "__main__":
    main()
