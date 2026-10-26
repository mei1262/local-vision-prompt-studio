"""Gradio 入口：视觉分析 + 提示词生成。"""

from __future__ import annotations

import hashlib
from typing import Any

import gradio as gr

from src.prompt.generator import generate_prompts
from src.vision.analyzer import analyze_image, attach_vlm

TARGET_MODELS = ["SDXL", "Flux", "视频生成（通用）"]


def _image_key(image: Any) -> str:
    raw = image.tobytes() if hasattr(image, "tobytes") else bytes(str(image.size), "utf-8")
    return hashlib.sha1(raw[: 256 * 1024] + raw[-4096:]).hexdigest()


def _candidate_label(item: dict[str, Any]) -> str:
    return f"{item['text']}  ({item['score']:.3f})"


def _candidate_labels(analysis: dict[str, Any]) -> list[str]:
    labels = []
    for item in analysis.get("scene_candidates") or []:
        if isinstance(item, dict) and item.get("text"):
            labels.append(_candidate_label(item))
    return labels


def _text_from_label(analysis: dict[str, Any], label: str) -> tuple[str, str]:
    for item in analysis.get("scene_candidates") or []:
        if isinstance(item, dict) and _candidate_label(item) == label:
            return str(item.get("text") or ""), str(item.get("mode") or "")
    return label, ""


def _radio_state(analysis: dict[str, Any]) -> tuple[Any, str]:
    labels = _candidate_labels(analysis)
    scene = str(analysis.get("scene_sentence") or "")
    value = labels[0] if labels else None
    for item in analysis.get("scene_candidates") or []:
        if isinstance(item, dict) and item.get("text") == scene:
            value = _candidate_label(item)
            break
    return gr.update(choices=labels, value=value), scene


def _format_outputs(analysis: dict[str, Any], result: dict[str, str]) -> tuple[str, str, str, str]:
    analysis_text = "\n".join(
        line
        for line in [
            (
                f"描述：{analysis.get('caption')}（{analysis.get('caption_verdict')}）"
                if analysis.get("caption")
                else f"描述：{analysis.get('caption_verdict') or '未启用'}"
            ),
            f"标签：{analysis.get('clip_content') or analysis.get('content', '')}",
            f"CLIP场面：{analysis.get('scene_sentence', '')}",
            f"VLM补充：{analysis.get('vlm_text') or analysis.get('vlm_note') or '（无）'}",
            f"媒介：{analysis.get('media_mode', '')}",
            f"风格标签：{', '.join(analysis.get('style_tags') or []) or '（空）'}",
            f"构图：{analysis.get('composition', '')}",
            f"色彩：{analysis.get('color', '')}",
            f"线条：{analysis.get('linework', '')}",
            f"备注：{analysis.get('notes', '')}",
            f"生成：{result.get('notes', '')}",
        ]
        if line
    )
    return (
        analysis_text,
        result.get("positive", ""),
        result.get("negative", ""),
        result.get("params", ""),
    )


def _for_prompt(analysis: dict[str, Any], use_vlm: bool) -> dict[str, Any]:
    copied = dict(analysis)
    if not use_vlm:
        copied["vlm_text"] = ""
        copied["vlm_note"] = "未调用 moondream（开关关闭）"
    return copied


def _empty(cache: dict[str, Any] | None) -> tuple:
    return (
        "请先上传一张图片。",
        "",
        "",
        "",
        cache or {},
        gr.update(choices=[], value=None),
        "",
    )


def _finish(
    analysis: dict[str, Any],
    target: str,
    extra: str,
    use_vlm: bool,
    cache: dict[str, Any],
) -> tuple:
    prompt_analysis = _for_prompt(analysis, bool(use_vlm))
    result = generate_prompts(
        analysis=prompt_analysis,
        target_model=target,
        extra_instruction=extra,
    )
    radio, scene = _radio_state(prompt_analysis)
    return (*_format_outputs(prompt_analysis, result), cache, radio, scene)


def run_pipeline(
    image: Any,
    target_model: str,
    extra_instruction: str,
    use_vlm: bool,
    cache: dict[str, Any] | None,
) -> tuple:
    if image is None:
        return _empty(cache)

    target = target_model or TARGET_MODELS[0]
    extra = extra_instruction or ""
    cache = dict(cache or {})
    key = _image_key(image)
    analysis = cache.get("analysis") if cache.get("key") == key else None

    if analysis is None:
        analysis = analyze_image(
            image,
            extra_instruction=extra,
            target_model=target,
            use_vlm=bool(use_vlm),
        )
    elif use_vlm and not analysis.get("vlm_attempted"):
        analysis = attach_vlm(analysis, image, extra, target)

    cache = {"key": key, "analysis": analysis}
    return _finish(analysis, target, extra, use_vlm, cache)


def apply_scene_sentence(
    scene_text: str,
    target_model: str,
    extra_instruction: str,
    use_vlm: bool,
    cache: dict[str, Any] | None,
) -> tuple:
    cache = dict(cache or {})
    analysis = cache.get("analysis")
    if not analysis:
        return (
            "请先点「分析并生成提示词」，再改场面句。",
            "",
            "",
            "",
            cache,
            gr.update(),
            scene_text or "",
        )
    analysis = dict(analysis)
    sentence = (scene_text or "").strip()
    if sentence:
        analysis["scene_sentence"] = sentence
        analysis["caption"] = sentence
        for item in analysis.get("scene_candidates") or []:
            if isinstance(item, dict) and item.get("text") == sentence and item.get("mode"):
                analysis["media_mode"] = item["mode"]
                break
    cache["analysis"] = analysis
    target = target_model or TARGET_MODELS[0]
    return _finish(analysis, target, extra_instruction or "", use_vlm, cache)


def on_pick_candidate(label: str, cache: dict[str, Any] | None) -> str:
    analysis = (cache or {}).get("analysis") or {}
    text, _mode = _text_from_label(analysis, label or "")
    return text


def on_vlm_toggle(
    use_vlm: bool,
    image: Any,
    target_model: str,
    extra_instruction: str,
    cache: dict[str, Any] | None,
) -> tuple:
    return run_pipeline(image, target_model, extra_instruction, bool(use_vlm), cache)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Local Vision Prompt Studio") as demo:
        cache = gr.State({})
        gr.Markdown(
            """
            # Local Vision Prompt Studio
            参考图 + 可选粗略意图 → 出图 / 出视频提示词。默认只跑 CLIP 和 3B。  
            分析后可在下面三句里点选，或直接改场面句，再点「按当前场面句生成」（不必重新跑 CLIP）。
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(label="输入图片", type="pil")
                target_model = gr.Dropdown(
                    label="目标生成模型",
                    choices=TARGET_MODELS,
                    value=TARGET_MODELS[0],
                )
                extra_instruction = gr.Textbox(
                    label="粗略意图 / 参考说明（可选）",
                    placeholder="你的粗略意图，例如：更冷清一点 / 缓慢推镜头 / 保留发型改成夜景",
                    lines=3,
                )
                use_vlm = gr.Checkbox(
                    label="使用 moondream 看图补充（慢，可能乱码；默认关）",
                    value=False,
                )
                run_btn = gr.Button("分析并生成提示词", variant="primary")
                scene_pick = gr.Radio(
                    label="CLIP 场面句候选（点选填入下方，再点生成）",
                    choices=[],
                    value=None,
                )
                scene_edit = gr.Textbox(
                    label="场面句（可编辑）",
                    lines=3,
                    placeholder="分析后会出现英文场面句，可改成你认为更准的一句",
                )
                apply_btn = gr.Button("按当前场面句生成")

            with gr.Column(scale=1):
                analysis_out = gr.Textbox(label="视觉分析", lines=10)
                positive_out = gr.Textbox(label="正向提示词", lines=6)
                negative_out = gr.Textbox(label="负向提示词", lines=4)
                params_out = gr.Textbox(label="参数建议", lines=3)

        outputs = [
            analysis_out,
            positive_out,
            negative_out,
            params_out,
            cache,
            scene_pick,
            scene_edit,
        ]
        run_btn.click(
            fn=run_pipeline,
            inputs=[image, target_model, extra_instruction, use_vlm, cache],
            outputs=outputs,
        )
        use_vlm.change(
            fn=on_vlm_toggle,
            inputs=[use_vlm, image, target_model, extra_instruction, cache],
            outputs=outputs,
        )
        scene_pick.change(
            fn=on_pick_candidate,
            inputs=[scene_pick, cache],
            outputs=[scene_edit],
        )
        apply_btn.click(
            fn=apply_scene_sentence,
            inputs=[scene_edit, target_model, extra_instruction, use_vlm, cache],
            outputs=outputs,
        )

    return demo


if __name__ == "__main__":
    build_ui().launch()
