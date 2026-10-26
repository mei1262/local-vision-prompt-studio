# 配置

| 文件 | 作用 |
| --- | --- |
| `vision.json` | CLIP 路径、阈值、DirectML / CPU 顺序 |
| `tag_vocab.json` | 零样本标签（主体、动作、风格等） |
| `scene_bank.json` | CLIP 场面句库；每条带 `mode`（photo / anime / graphic / paint） |
| `llm.json` | Ollama 地址、3B 润色模型；`vlm_enabled` 控制是否调用 moondream（默认 false） |
| `caption.json` | 旧的 ViT-GPT2 描述，`enabled` 为 false，不要打开 |

改 `tag_vocab.json` 或 `scene_bank.json` 后执行：

```powershell
python scripts/rebake_tag_embeddings.py
```

再重启界面。不重算的话，分析仍用磁盘上的旧向量。

权重不在 `configs/` 里：CLIP 在项目 `models/`，Ollama 在用户目录 `.ollama/models/`。完整对照见仓库根目录 [MODELS.md](../MODELS.md)。
