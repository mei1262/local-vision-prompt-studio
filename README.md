# Local Vision Prompt Studio

[English](#english) · [中文](#中文)

本地提示词工作台：参考图 + 可选粗略意图 → 更适合喂给 AI **出图 / 出视频** 的提示词。  
本机不出成品图、不渲长片。

---

## 中文

### 做什么

把一张参考图（实拍、插画、封面、截图）分析成可编辑的英文场面句和标签，再拼出正向 / 负向提示词和简单参数，复制到 SDXL、Flux 或外部视频工具里用。

- **不是** 一张图自动生成完整电影分镜。
- **不是** 在本机跑扩散或视频生成。
- 可选粗略意图（例如「更冷清」「缓慢推镜头」）只影响写法，不代替看图。

### 现在能用的

- CLIP ViT-L/14（ONNX + DirectML，失败则 CPU / ViT-B/16）标签 + 场面句检索
- 场面句三候选，可点选、可手改后再生成
- 按选定场面句丢掉明显冲突的标签（例如单人场面里的「三人」「胶片截帧」）
- 目标档：SDXL / Flux / 视频（通用，目前偏简单）
- 本机 Ollama `qwen2.5:3b` 润色（可选）
- `moondream` 看图开关（默认关；本机常乱码且慢）

型号取舍见 [MODELS.md](MODELS.md)，开发进度见 [PROJECT.md](PROJECT.md)，下一步取舍见 [ROADMAP.md](ROADMAP.md)。

### 硬件约定

面向笔记本：**AMD Ryzen 7 7840HS + Radeon 780M，16GB 内存**。离线优先，磁盘尽量 10–15GB。核显走 DirectML，不要默认假设有 NVIDIA / CUDA。

### 快速开始

需要 Python 3.10+。请用项目 `.venv`，不要装进 Anaconda `base`。

```powershell
git clone https://github.com/mei1262/local-vision-prompt-studio.git
cd local-vision-prompt-studio
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python scripts/prepare_vision_assets.py --source mirror --no-proxy
python app.py
```

Linux / macOS 把激活改成 `source .venv/bin/activate`。没有 DirectML 时会走 CPU。

国内直连 Hugging Face 常失败，准备脚本默认先走 [hf-mirror.com](https://hf-mirror.com)。若设置了代理却没开，加上 `--no-proxy`。

**只装 `onnxruntime-directml`，不要再装官方 `onnxruntime`。**

可选润色（需 [Ollama](https://ollama.com)）：

```text
ollama pull qwen2.5:3b
```

可选看图（默认关，体积约 1.7GB）：

```text
ollama pull moondream
```

改过 `configs/tag_vocab.json` 或 `configs/scene_bank.json` 之后：

```powershell
python scripts/rebake_tag_embeddings.py
```

然后重启 `python app.py`。

仓库**不含** CLIP / Ollama 权重。CLIP 下到项目 `models/`，Ollama 模型在用户目录 `.ollama/`。

### 许可

代码以 MIT 许可。第三方模型（CLIP、Qwen、Moondream 等）遵守各自原始许可证。

---

## English

### What this is

A **local prompt workbench**: reference image + optional rough intent → prompts you can paste into AI **image** or **video** generators.

It does **not** render images or videos. It does **not** turn one still into a full movie storyboard.

### What works today

- CLIP ViT-L/14 (ONNX + DirectML, with CPU / ViT-B/16 fallback) for tags and scene-sentence retrieval
- Top-3 scene candidates; pick or edit, then regenerate
- Drop tags that contradict the chosen scene (e.g. “three people” on a single-subject shot)
- Targets: SDXL, Flux, and a simple generic video template
- Optional Ollama `qwen2.5:3b` rewrite
- Optional `moondream` (off by default; often noisy and slow on this hardware)

See [MODELS.md](MODELS.md) for model choices, [PROJECT.md](PROJECT.md) for status, [ROADMAP.md](ROADMAP.md) for what is worth building next.

### Hardware

Aimed at a laptop: **Ryzen 7 7840HS + Radeon 780M, 16GB RAM**. Offline-first. Rough disk budget 10–15GB. Use DirectML on the iGPU; do not assume NVIDIA CUDA.

### Quick start

Python 3.10+. Use the project `.venv`, not Anaconda `base`.

```powershell
git clone https://github.com/mei1262/local-vision-prompt-studio.git
cd local-vision-prompt-studio
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python scripts/prepare_vision_assets.py --source auto --no-proxy
python app.py
```

On Linux/macOS: `source .venv/bin/activate`. Without DirectML, inference falls back to CPU.

Install **only** `onnxruntime-directml`, not stock `onnxruntime` at the same time.

Optional:

```text
ollama pull qwen2.5:3b
ollama pull moondream
```

After editing the tag vocab or scene bank:

```powershell
python scripts/rebake_tag_embeddings.py
```

Weights are **not** in git. CLIP goes under `models/`; Ollama models live in `~/.ollama/` (Windows: `%USERPROFILE%\.ollama\`).

### License

Code: MIT. Third-party weights keep their own licenses.
