# Local Vision Prompt Studio

[English](#english) · [中文](#中文)

本地提示词工作台：参考图 + 可选意图 → 复制给外部模型去出**新图 / 新视频**。  
本仓库不渲染成品，也不把原图再生成一遍。

---

## 中文

### 这是什么

把一张「本镜」理解成：一句可编辑的英文场面句 + 若干英文标签，再拼成正向 / 负向提示词和简单参数。你把结果粘到 SDXL、Flux 或外部视频工具里，去生成**另一张新图**（或一条视频提示）。

它**不是**：本机扩散出图、本机渲视频、自动写电影分镜、把参考图 1:1 复刻。

### 两套「库」，不要混

| 库 | 文件 | 里面是什么 | 怎么用 |
| --- | --- | --- | --- |
| 场面句库 | `configs/scene_bank.json` | **完整英文句子**，每条是一个整场面（例如「一张单人看镜头的照片」） | CLIP 整句检索，选出最像的 3 句。**不会**把「和服 + 背影 + 少年」拆开再拼成一句 |
| 标签词表 | `configs/tag_vocab.json` | **零散元素**：主体、人数、景别、构图、风格、光线、颜色等 | 按组打分，挑中的短语叠进提示词 |

最终提示词 ≈ **你选定（或手改）的那句场面句** + **标签短语** + **可选的角色/风格锁** + **粗略意图**。

所以：「和服、背影、侧面、少年」这类碎片在**标签层**（以及你自己改的场面句 / 档案锁词）里起作用，不在场面句库里当积木。场面句库必须是能单独成立的整句，而且句与句要差得开；太像的整句会互相抢分。

场面句库大约 **257 条通用整句**，按媒介和场面类型铺开（人数、朝向、室内外、风景、动物、器物、绘画、平面），**不按某几张常用参考图来写**。句与句在主体或地点上分开，避免近义特写互相抢分。对不准时：在三个候选里点另一句，或直接改场面句——句库仍然不可能穷尽每张图。

### 已知限制

分析一张图时，CLIP 把图编成一个向量，和这 257 句**事先算好的文本向量**逐条做相似度，取得分最高的三句。句子在点「分析」时不会被重读。改句库后要跑 `scripts/rebake_tag_embeddings.py` 才会更新这些向量。

CLIP 能把配对过的图文拉到附近，所以「两个人并排」有时会比「森林」更靠近一张双人剪影。它**不读句子、不数人、不认服装**。分数多在 0.10–0.22，只表示相对远近；第一名和第二名可以只差 0.005，第一名也可以是错的（例如把人物排成「野外哺乳动物」，或把日落双人影排到「棚拍背景」附近）。

加句只增加可比的落点，不会让模型更会看图。标签词表是另一套大约一百个封闭词，仍可能和场面句打架；冲突过滤只丢掉已知的相反短语。`qwen2.5:3b` 只改写已有文字，看不懂图，可能把一句错的场面句写得很顺。`moondream` 默认关闭。角色/风格锁是往提示里加短语，不能保证换构图后还是同一张脸。

视频侧一次写一个镜头。参考图另交视频模型。正文按模板套：在什么环境里，什么人物，在什么音乐下，做什么动作，再加上景别、运镜、光线、焦点、画质和时长。不写对白，不拆成下一镜。

### 特点

- 离线、轻量：ONNX CLIP（Windows 优先 DirectML，否则 CPU）看图；可选 Ollama 只做文字润色
- 本镜管**新构图**；角色/风格参考和档案只锁外貌或画风
- 场面句三选一，可手改后再生成，不必重跑 CLIP
- 选定场面句后丢掉明显冲突的标签（例如单人句里的「三人」）
- 复制单栏或「复制套装」；历史只存提示，不存原图
- 页面上用「图片侧 / 视频侧」分开。图片侧给 SDXL / Flux；视频侧单独写一条视频提示（镜头、运动、时长来自选项）
- `moondream` 看图默认关（慢，且经常乱码）

### 界面怎么用

1. **本镜**：上传要分析构图的那张图（未来要出的是新图，不是这张的复制）。
2. **角色参考 / 风格参考**（可选）：最多 3 张锁外貌、2 张锁画风。不填也能用。
3. **目标生成模型**：SDXL 偏逗号标签；Flux 偏短句；视频档会加一点镜头运动词。
4. **粗略意图**（可选）：只影响写法，例如「更冷清」「换成夜景」「缓慢推镜头」。不代替看图。
5. **本次套用的档案**（可选）：见下文。空着不影响分析。
6. 保持 **moondream 关闭**，除非你想试看图补充。
7. 点 **分析并生成提示词**。
8. 看左边三个场面句候选：点一句会填入编辑框；不对就**手改成你认为准的一句英文**。
9. 点 **按当前场面句生成**（改句后用这个，不必重新分析）。
10. 右边复制正向 / 负向 / 参数，或复制套装，贴到外部生成器。

**角色 / 风格档案不用填。** 只有「同一张脸或同一套画风，反复出新构图」时才值得存。可手写英文锁词，或从参考图点「提取草稿」，删掉构图词再保存，生成时勾选套用。

**状态说明不是历史。** 点「分析并生成」或「按当前场面句生成」成功后，会自动在「历史」里记一条提示。「再存一条」只是把当前提示再存一次。载入历史只恢复词，不会把原图重新喂进去。

历史和档案在本地 `outputs/workbench/`，不进 git。

### 环境与安装

Python 3.10+。用项目 `.venv`，不要装进 Anaconda `base`。Windows 上只装 `onnxruntime-directml`，不要再装官方 `onnxruntime`。仓库不含权重。

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

Linux / macOS：`source .venv/bin/activate`，准备脚本可用 `--source auto`。国内直连 Hugging Face 常失败，脚本默认走 [hf-mirror.com](https://hf-mirror.com)；代理没开时加 `--no-proxy`。

可选润色：

```text
ollama pull qwen2.5:3b
```

改过 `configs/tag_vocab.json` 或 `configs/scene_bank.json` 之后必须重算向量，否则界面仍用旧库：

```powershell
python scripts/rebake_tag_embeddings.py
```

自己加场面句时：写**完整英文句子**，带 `id` / `mode`（`photo` `anime` `paint` `graphic`）/ `en` / `zh`。不要加和现有句几乎一样的近义句，也不要按某一张常用图特写服装或角色。标签才适合加「和服」这类元素。

CLIP 在项目 `models/`。Ollama 模型在用户目录 `.ollama/`。

更细的型号说明见 [MODELS.md](MODELS.md)，配置见 [configs/README.md](configs/README.md)。

### 许可

代码 MIT。CLIP / Qwen / Moondream 等权重遵守各自许可证。

---

## English

### What this is

A local workbench: one **shot image** + optional intent → an editable English scene sentence, tags, and positive/negative prompts you paste into **new** SDXL / Flux / video generations.

It does **not** render images or video, write a movie storyboard, or recreate the reference 1:1.

### Two libraries

| Library | File | Contents | Role |
| --- | --- | --- | --- |
| Scene bank | `configs/scene_bank.json` | **Complete English sentences**, each a whole scene | CLIP retrieves the nearest 3 **as wholes**. It does **not** snap fragments like “kimono + back view + boy” into one sentence |
| Tag vocab | `configs/tag_vocab.json` | **Atomic labels**: subject, count, shot, composition, style, light, color | Scored per group and appended to the prompt |

Final prompt ≈ **chosen (or edited) scene sentence** + **tag phrases** + **optional identity/style locks** + **intent**.

The scene bank is about **257 generic full sentences** across media and scene types. It is **not** written around a handful of personal reference images. When retrieval misses, pick another candidate or edit the sentence.

### Known limits

CLIP encodes the image once and scores it against **precomputed** vectors for every scene sentence. It does not reread the sentences at click time. After editing the bank, run `scripts/rebake_tag_embeddings.py`.

Scores usually sit around 0.10–0.22 and only rank neighbors. The winner can be wrong by a few thousandths (a person ranked as “a wild mammal”, a sunset two-shot near “studio backdrop”). More sentences add more neighbors; they do not add comprehension. The tag list is a separate closed set and can still contradict the scene. The 3B rewriter cannot see the image. Profile locks are extra phrases, not a guarantee of the same face.

The video page writes one shot. The reference image is sent to the video model separately. The text fills a template: place, subject, music, and action, then shot size, camera, light, focus, grade, and duration. No dialogue and no second shot.

### Features

- Offline CLIP (ONNX; DirectML on Windows when available, else CPU)
- Optional Ollama rewrite; `moondream` off by default
- Shot image owns composition; extra refs/profiles only lock look or style
- Top-3 scenes, editable, regenerate without re-running CLIP
- Conflict tags dropped after you commit a scene
- Copy fields + paste pack; history stores prompts only
- A page switch separates still prompts (SDXL / Flux) from a single video prompt (camera, motion, and duration are selected, not inferred)

### How to use

1. Upload the **shot** (composition for a *new* picture).
2. Optionally add character refs (look) and style refs (medium/light).
3. Pick SDXL / Flux / video; optional intent; optional profiles. Leave moondream off unless experimenting.
4. **Analyze and generate**.
5. Pick or edit the scene sentence, then **Generate from current scene**.
6. Copy prompts into an external generator.

Profiles are optional. History is saved automatically after a successful generate; the status line is only a log of what just happened.

After editing the vocab or scene bank, run `python scripts/rebake_tag_embeddings.py` and restart the app. Add **full sentences** to the scene bank, not near-duplicates and not one-off outfits from a favorite image. Put elements such as clothing in the tag vocab.

### Install

Python 3.10+. Project `.venv` only. On Windows install **only** `onnxruntime-directml`.

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

Weights are not in git. CLIP → `models/`. History/profiles → `outputs/workbench/`. Ollama → `~/.ollama/`.

See [MODELS.md](MODELS.md) and [configs/README.md](configs/README.md).

### License

Code: MIT. Third-party weights keep their own licenses.
