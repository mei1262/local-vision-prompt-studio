"""下载轻量 caption ONNX（优先 ViT-GPT2；BLIP 仓库缺文件则跳过）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.captioner import load_caption_config
from src.vision.hf_download import disable_proxies, download_repo_file, resolve_endpoints
from src.vision.paths import resolve_path


def _try_names(repo: str, names: list[str], dest: Path, endpoints: list[str]) -> None:
    last_error: Exception | None = None
    for name in names:
        try:
            download_repo_file(repo, name, dest, endpoints)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"  换文件名：{name} 失败（{exc}）")
    raise RuntimeError(f"{repo} 找不到编码器/解码器") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="下载图像描述 ONNX。")
    parser.add_argument("--source", choices=("auto", "mirror", "official"), default="auto")
    parser.add_argument("--no-proxy", action="store_true")
    args = parser.parse_args()
    if args.no_proxy:
        disable_proxies()
        print("已忽略代理环境变量。")

    endpoints = resolve_endpoints(args.source)
    print("下载端点：" + " → ".join(endpoints))
    config = load_caption_config()
    ok = 0
    for model in config["models"]:
        print(f"\n=== {model['label']} ===")
        try:
            _try_names(
                model["repo"],
                [model["remote_encoder"], "onnx/encoder_model_quantized.onnx", "onnx/encoder_model.onnx"],
                resolve_path(model["encoder"]),
                endpoints,
            )
            _try_names(
                model["repo"],
                [model["remote_decoder"], "onnx/decoder_model_quantized.onnx", "onnx/decoder_model.onnx"],
                resolve_path(model["decoder"]),
                endpoints,
            )
            download_repo_file(
                model["repo"],
                model["remote_tokenizer"],
                resolve_path(model["tokenizer"]),
                endpoints,
            )
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"跳过 {model['id']}：{exc}")
    if not ok:
        raise RuntimeError("没有下载到任何 caption 模型。")
    print(f"\n完成，可用模型 {ok} 个。之后分析时会自动用 caption 描述画面。")


if __name__ == "__main__":
    main()
