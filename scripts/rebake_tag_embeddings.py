"""词表变更后，用已有文本 ONNX 重算标签向量。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.embeddings import bake_scene_embeddings, bake_tag_embeddings
from src.vision.paths import load_vision_config, resolve_path
from src.vision.runtime import available_providers, create_session


def main() -> None:
    config = load_vision_config()
    providers = available_providers()
    provider = "CPUExecutionProvider" if "CPUExecutionProvider" in providers else providers[0]
    print(f"重算标签向量，执行器：{provider}")
    for model in config["models"]:
        text_path = resolve_path(model["text_onnx"])
        tokenizer_path = resolve_path(model["tokenizer"])
        embed_path = resolve_path(model["embeddings"])
        if not text_path.is_file() or not tokenizer_path.is_file():
            print(f"跳过 {model['label']}：缺少文本塔或 tokenizer")
            continue
        print(f"计算 {model['label']} …")
        session = create_session(text_path, provider)
        try:
            bake_tag_embeddings(session, tokenizer_path, embed_path)
            if model.get("scene_embeddings"):
                bake_scene_embeddings(session, tokenizer_path, resolve_path(model["scene_embeddings"]))
                print(f"已写入场面向量 {model['scene_embeddings']}")
        finally:
            del session
        print(f"已写入 {embed_path}")


if __name__ == "__main__":
    main()
