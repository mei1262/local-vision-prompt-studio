# Local Vision Prompt Studio

**目的：** 你提供参考图和可选的粗略意图，工具补成更适合喂给 AI 出图 / 出视频的提示词。不是一张图自动生成完整分镜本，也不在本机渲长片。SDXL / Flux / 视频模型在外部使用。

本地运行：上传一张图片（动漫 / 插画 / 普通照片均可），用 CLIP 做标签和场面检索，再写成提示词。Ollama 3B 只负责润色，不负责看图。型号为什么这样选、东西下到了哪，见 [MODELS.md](MODELS.md)。

目标机器：AMD Ryzen 7 7840HS + 780M 核显，16GB 内存。全部离线，视觉走 ONNX + DirectML，语言走 Ollama 3B–4B，界面用 Gradio。磁盘尽量控制在 10–15GB 以内。ASCII / 线稿当前不做。

## 现在怎么工作

视觉只有 **一套模型：CLIP ViT-L/14**（失败则 CPU，再失败则 ViT-B/16）。它不生成文字，只比较图像向量和预写好的英文：

| 输出 | 来源 | 说明 |
| --- | --- | --- |
| 描述 / 场面句 | `configs/scene_bank.json` | 在固定场面句里检索最像的一句，写入正向提示词 |
| 标签 | `configs/tag_vocab.json` | 主体、动作、风格、光线、构图等词 |
| 媒介 | 规则 + 场面句的 `mode` | photo / anime / graphic / paint |
| 色彩 / 构图 / 线条 | Pillow 规则，不占模型体积 | 辅助，不是语义 |
| 正向 / 负向 / 参数 | `src/prompt/assembler.py` | 场面句 + 标签模板 |
| 润色 | 本机 Ollama `qwen2.5:3b` | 可选；看不懂图，只能改写已有句子 |

ViT-GPT2 那一刀已关闭并删除权重：它是另一个小生成模型，会编造，和 CLIP 不是同一套理解。

## 目前已完成

- 小 VLM：走本机 Ollama `moondream`（约 2B）。看参考图 + 粗略意图，写出补充描述；未 pull 时仍只用 CLIP。
- 场面检索：同一套 CLIP 在句库里选描述。实测有效的例子：打电话的动漫角色、黑白理发、摄像头自拍。
- 第一刀：冲突标签去重、照片/动漫规则、动作标签（打电话 / 理发 / 摆拍）、按构图出参数。
- 提示词：规则模板 + Ollama（`127.0.0.1:11434`，本机已有 `qwen2.5:3b`）。
- 资产：
  - `models/clip-vit-l-14/` 主力（含 `tag_embeddings.npz`、`scene_embeddings.npz`）
  - `models/clip-vit-b-16/` 回退
- 国内下载：`hf-mirror.com`、续传、`--no-proxy`。
- 依赖在项目 `.venv`（Gradio 5.50、onnxruntime-directml）。

改词表或场面句库后必须重算向量，否则界面仍用旧句：

```powershell
python scripts/rebake_tag_embeddings.py
```

然后重启 `python app.py`。

日志里的 `Some nodes were not assigned to the preferred execution providers` 是 ONNX Runtime 正常提示。

## 能力对照

| 能力 | 状态 |
| --- | --- |
| 上传并分析 | 已可用 |
| 场面描述（CLIP 检索） | 已可用；句库里有的场面明显变准 |
| 标签 | 已可用；仍会和场面句打架（例如理发图仍标专辑封面） |
| 正向 / 负向 / 参数 | 已可用 |
| SDXL / Flux / 视频模板 | 已接三套 |
| 追加指令 | 规则关键词 + Ollama 润色 |
| 任意画面自由描述 | 做不到。CLIP 只能在句库里选，不会写新句子 |
| 人数 / 朝向 / 细物件 | 仍不可靠 |

## 下一步

场面三候选、可编辑场面句、按场面句丢掉冲突标签（三人/对镜/胶片截帧等）已接上。moondream 仍是可选开关。

## 场面优先（部分已完成）

目标：保持现在这套 CLIP，把「场面句已经对、提示词里还夹着错标签」清掉，并按测过的图补句库。不换大视觉模型。

### 1. 场面句压过冲突标签

当 CLIP 已经选出「黑白理发」时，正向提示词里不该再出现 `album cover`、`graphic design`。当选出「看着镜头」时，不该再出现 `looking away`。

做法：在 `src/prompt/assembler.py` 里按场面句和 `scene_bank` 的 `mode` 建一份禁用表（封面/海报 vs 理发，对镜两人 vs 单人，自拍 vs 街景等），组装前丢掉相反的标签。Ollama 提示里写明：场面句是事实，与它矛盾的标签全部丢掉。

### 2. 按测过的图补句库

`configs/scene_bank.json` 是封闭集合。没写进去的场面会选最近似的一句。补句原则：一句只说一个可核对的事实，带媒介（anime / photo / graphic），避免空泛的 mood。

每加句后执行 `rebake_tag_embeddings.py` 并重启。界面备注里的「候选」三句是调库依据：第二名经常抢戏就改措辞或加对照句。

### 3. 先不要做的

- 再开 ViT-GPT2 / BLIP。
- Florence-2、Qwen2-VL、LLaVA（体积和内存超出现阶段约定）。
- 云端 API、多用户、ASCII。

句库覆盖常见类型、错标签不再进提示词之后，再考虑导出复制和历史记录。

## 本地怎么跑

```powershell
cd local-vision-prompt-studio
.\.venv\Scripts\Activate.ps1
python app.py
```

换图后必须再点一次「分析并生成提示词」。Ollama 托盘保持运行才会润色。
