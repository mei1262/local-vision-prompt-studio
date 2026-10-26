from __future__ import annotations

import os
import time
from pathlib import Path

import requests

OFFICIAL_ENDPOINT = "https://huggingface.co"
MIRROR_ENDPOINT = "https://hf-mirror.com"


def resolve_endpoints(choice: str = "auto") -> list[str]:
    env_endpoint = (os.environ.get("HF_ENDPOINT") or "").rstrip("/")
    if choice == "official":
        return [OFFICIAL_ENDPOINT]
    if choice == "mirror":
        return [MIRROR_ENDPOINT]
    endpoints: list[str] = []
    if env_endpoint:
        endpoints.append(env_endpoint)
    endpoints.extend([MIRROR_ENDPOINT, OFFICIAL_ENDPOINT])
    unique: list[str] = []
    for item in endpoints:
        if item not in unique:
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
                if response.status_code == 404:
                    raise FileNotFoundError(f"404 {url}")
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
        except FileNotFoundError:
            raise
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
        except FileNotFoundError as exc:
            errors.append(f"{endpoint}: {exc}")
            print(f"  404，跳过其余端点：{filename}")
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{endpoint}: {exc}")
            print(f"切换端点重试（当前失败：{endpoint}）")
    raise RuntimeError("无法下载：\n" + "\n".join(errors))
