"""根据视觉分析结果生成提示词：有 Ollama 则改写，否则用规则组装。"""

from __future__ import annotations

from typing import Any

from src.llm.ollama_client import generate_json, is_available, pick_model, usable_vlm_text
from src.prompt.assembler import assemble_prompts, suggest_params


def _build_llm_prompt(
    analysis: dict[str, Any],
    target_model: str,
    extra_instruction: str,
    draft: dict[str, str],
) -> str:
    return f"""You rewrite image prompts. Return JSON only:
{{"positive":"...","negative":"...","params":"..."}}
Rules:
- English only in positive/negative.
- Refine a generation prompt. Do not write a multi-shot storyboard or movie script.
- The CLIP scene is the safe description. Use VLM notes only if they are normal English sentences.
- Ignore VLM notes that look like "ids", numbers-only, or garbled characters.
- Do not invent people or objects. English only in positive/negative. No CJK in prompts.
- Use the user intent as direction (style, mood, motion), not as new objects.
- If the user intent conflicts with the image, keep the image and adapt the wording.
- Do not keep conflicting tags (half-body and full-body, anime and photoreal).
- Do not add people-count, mirror, album cover, mecha, film still, or child unless they appear in the scene sentence.
- If media_mode is photo, never write anime screencap.
- If media_mode is anime, never write photorealistic photograph.
- Negative must oppose media_mode.
- Keep the suggested resolution in params.
Target model: {target_model}
SDXL: comma-separated tags.
Flux: 1-3 natural sentences.
Video: add camera and motion.

Analysis:
media_mode: {analysis.get("media_mode", "")}
scene: {analysis.get("scene_sentence", "")}
vlm_visual_notes: {usable_vlm_text(str(analysis.get("vlm_text") or "")) or "(none)"}
style_tags: {", ".join(analysis.get("style_tags") or [])}
composition: {analysis.get("composition", "")}
color: {analysis.get("color", "")}
linework: {analysis.get("linework", "")}

User extra instruction: {extra_instruction or "(none)"}

Draft you may improve:
positive: {draft.get("positive", "")}
negative: {draft.get("negative", "")}
params: {draft.get("params", "")}
"""


def generate_prompts(
    analysis: dict[str, Any],
    target_model: str,
    extra_instruction: str = "",
) -> dict[str, str]:
    """根据分析结果生成提示词。"""
    draft = assemble_prompts(analysis, target_model, extra_instruction)
    notes = "提示词来源：规则模板。"

    if is_available():
        model = pick_model()
        if model:
            improved = generate_json(
                _build_llm_prompt(analysis, target_model, extra_instruction, draft),
                model=model,
            )
            if improved and improved.get("positive"):
                draft["positive"] = str(improved["positive"]).strip()
                if improved.get("negative"):
                    draft["negative"] = str(improved["negative"]).strip()
                if improved.get("params") and "分辨率" in str(improved["params"]):
                    draft["params"] = str(improved["params"]).strip()
                else:
                    draft["params"] = suggest_params(
                        analysis,
                        target_model,
                        str(analysis.get("media_mode") or "general"),
                    )
                notes = f"提示词来源：Ollama（{model}）润色。"
            else:
                notes = f"Ollama 在线（{model}），但改写失败，已回退规则模板。"
        else:
            notes = (
                "Ollama 在运行，但没有可用模型。可执行：ollama pull qwen2.5:3b"
            )
    else:
        notes = (
            "未检测到 Ollama，已用规则模板生成可用提示词。"
            "安装后执行 ollama pull qwen2.5:3b 可提升改写质量。"
        )

    draft["notes"] = notes
    draft["analysis_preview"] = str(analysis.get("content", ""))
    return draft
