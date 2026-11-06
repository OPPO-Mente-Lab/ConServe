# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

"""SP/UP (system_prompt / user_prompt) token counting and JSON output fields -- shared by the vLLM step1 scripts."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

ImageLayout = Literal["crop_only", "dual_image", "fullpage_single"]


def _get_tokenizer(processor: Any) -> Any:
    return getattr(processor, "tokenizer", processor)


def _encode_len(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(text))


def _apply_template(
    processor: Any,
    messages: list[dict[str, Any]],
    apply_chat_template: Callable[..., str] | None,
    **kwargs: Any,
) -> str:
    fn = apply_chat_template or processor.apply_chat_template
    try:
        return fn(processor, messages, tokenize=False, **kwargs)
    except TypeError:
        return fn(messages, tokenize=False, **kwargs)


def _apply_template_sp_baseline(
    processor: Any,
    system_prompt: str,
    apply_chat_template: Callable[..., str] | None,
    **kwargs: Any,
) -> str:
    """The Qwen chat_template requires a user message; try minimal user placeholders such as empty string/space in order."""
    last_err: Exception | None = None
    for user_stub in ("", " ", "."):
        try:
            return _apply_template(
                processor,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_stub},
                ],
                apply_chat_template,
                **kwargs,
            )
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "no user query" in msg or "templateerror" in type(e).__name__.lower():
                continue
            raise
    raise RuntimeError(
        f"无法渲染 SP baseline chat_template: {last_err}"
    ) from last_err


def count_sp_up_tokens(
    processor: Any,
    system_prompt: str,
    user_prompt: str,
    *,
    num_images: int = 1,
    apply_chat_template: Callable[..., str] | None = None,
) -> tuple[int, int, int]:
    """Count SP / UP / total text-template tokens with the same Qwen tokenizer + chat_template as vLLM.

    Returns:
        (system_prompt_tokens_sp, user_prompt_text_tokens_up, prompt_text_tokens_template)
        ``prompt_text_tokens_template`` includes template special tokens and image
        placeholders but **not** the expanded visual tokens; visual tokens are
        approximately ``n_prompt - prompt_text_tokens_template``.
    """
    tokenizer = _get_tokenizer(processor)
    gen_kw: dict[str, Any] = {"add_generation_prompt": True, "enable_thinking": False}

    # The Qwen3-VL chat_template requires messages to contain a user message; system alone is not allowed.
    # SP = template tokens of system + the minimal user placeholder; UP = the increment after filling in the real user_prompt.
    text_sp_base = _apply_template_sp_baseline(
        processor, system_prompt, apply_chat_template, **gen_kw,
    )
    sp_tokens = _encode_len(tokenizer, text_sp_base)

    text_sp_up = _apply_template(
        processor,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        apply_chat_template,
        **gen_kw,
    )
    sp_up_tokens = _encode_len(tokenizer, text_sp_up)
    up_tokens = max(0, sp_up_tokens - sp_tokens)

    user_content: Any
    if num_images <= 0:
        user_content = user_prompt
    else:
        user_content = [
            *[{"type": "image", "image": "__PLACEHOLDER__"} for _ in range(num_images)],
            {"type": "text", "text": user_prompt},
        ]
    text_with_placeholders = _apply_template(
        processor,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        apply_chat_template,
        **gen_kw,
    )
    template_tokens = _encode_len(tokenizer, text_with_placeholders)
    return sp_tokens, up_tokens, template_tokens


def estimate_text_tokens_tiktoken(text: str, encoding_name: str = "cl100k_base") -> int:
    """Rough count with GPT tiktoken (does not match Qwen's tokenization; used only as a fallback when no processor is available)."""
    try:
        import tiktoken  # type: ignore  # noqa: WPS433

        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text or ""))
    except Exception:
        return max(0, len(text or "") // 2)


def _count_sp_up(
    *,
    processor: Any | None,
    system_prompt: str,
    user_prompt: str,
    num_images: int,
    apply_chat_template: Callable[..., str] | None,
    n_prompt: int,
) -> tuple[int, int, int, str]:
    if processor is not None:
        sp, up, tpl = count_sp_up_tokens(
            processor,
            system_prompt,
            user_prompt,
            num_images=num_images,
            apply_chat_template=apply_chat_template,
        )
        return sp, up, tpl, "qwen_tokenizer+chat_template"

    sp_raw = estimate_text_tokens_tiktoken(system_prompt)
    up_raw = estimate_text_tokens_tiktoken(user_prompt)
    total_raw = sp_raw + up_raw
    budget = max(0, int(n_prompt) - 1)
    if total_raw > budget and total_raw > 0:
        sp = max(0, int(round(sp_raw * budget / total_raw)))
        up = max(0, budget - sp)
        tpl = sp + up
    else:
        sp, up, tpl = sp_raw, up_raw, total_raw
    return sp, up, tpl, "tiktoken_cl100k_base_capped"


def split_image_tokens_by_pixels(
    total_image_tokens: int,
    global_wh: tuple[int, int],
    crop_wh: tuple[int, int],
) -> tuple[int, int]:
    """When two images share a single total image-token count, split it between full image / crop proportionally to pixel area."""
    gw, gh = max(1, global_wh[0]), max(1, global_wh[1])
    cw, ch = max(1, crop_wh[0]), max(1, crop_wh[1])
    gpx = gw * gh
    cpx = cw * ch
    s = gpx + cpx
    full_t = int(round(total_image_tokens * (gpx / s)))
    crop_t = total_image_tokens - full_t
    return full_t, crop_t


def build_vlm_token_cost(
    *,
    n_prompt: int,
    n_completion: int,
    system_prompt: str,
    user_prompt: str,
    global_wh: tuple[int, int],
    crop_wh: tuple[int, int],
    image_layout: ImageLayout = "crop_only",
    processor: Any | None = None,
    apply_chat_template: Callable[..., str] | None = None,
    num_images: int | None = None,
) -> dict[str, Any]:
    """Aligned with the ``doubao_token_cost`` keys of the Doubao step scripts so downstream can parse them uniformly."""
    if num_images is None:
        num_images = 2 if image_layout == "dual_image" else 1

    sp, up, text_tpl, source = _count_sp_up(
        processor=processor,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        num_images=num_images,
        apply_chat_template=apply_chat_template,
        n_prompt=n_prompt,
    )
    sp = min(sp, int(n_prompt))
    up = min(up, max(0, int(n_prompt) - sp))
    text_tpl = min(text_tpl, int(n_prompt))
    image_residual = max(0, int(n_prompt) - text_tpl)

    if image_layout == "dual_image":
        image_full, image_crop = split_image_tokens_by_pixels(
            image_residual, global_wh, crop_wh,
        )
        split_note = (
            "视觉 token = vLLM prompt_token_ids 减 chat_template 文本(含 image 占位符); "
            "再按 global/crop 像素面积比拆分。"
        )
    elif image_layout == "fullpage_single":
        image_full, image_crop = image_residual, 0
        split_note = "视觉 token = vLLM prompt 减 chat_template 文本(含 image 占位符)。"
    else:
        image_full, image_crop = 0, image_residual
        split_note = "视觉 token = vLLM prompt 减 chat_template 文本(含 image 占位符)。"

    legend = (
        "vLLM: prompt/completion 为引擎 token id 计数; "
        "SP/UP 为 Qwen tokenizer + chat_template 口径(与 prompt_tokens 一致的分词器)."
        if source == "qwen_tokenizer+chat_template"
        else "vLLM: prompt/completion 为引擎计数; SP/UP 为 tiktoken 回退并已 cap 到 prompt 以内."
    )

    return {
        "input_prompt_tokens_total": int(n_prompt),
        "output_completion_tokens": int(n_completion),
        "system_prompt_tokens_sp": sp,
        "user_prompt_text_tokens_up": up,
        "prompt_text_tokens_template": text_tpl,
        "estimated_system_prompt_tokens_sp": sp,
        "estimated_user_prompt_text_tokens_up": up,
        "estimated_text_tokens_sp_plus_up": sp + up,
        "token_count_source": source,
        "prompt_text_tokens_api": None,
        "prompt_image_tokens_api_total": None,
        "prompt_image_tokens_inferred_total": image_residual,
        "prompt_cached_tokens_api": None,
        "image_tokens_fullpage": image_full,
        "image_tokens_crop": image_crop,
        "image_tokens_split_note": split_note,
        "global_context_wh": [global_wh[0], global_wh[1]],
        "crop_wh": [crop_wh[0], crop_wh[1]],
        "usage_raw": {},
        "_legend": legend,
    }


def sum_token_cost_field(paragraphs_out: list[dict[str, Any]], key: str) -> int:
    total = 0
    for p in paragraphs_out:
        tc = p.get("doubao_token_cost")
        if not isinstance(tc, dict):
            continue
        v = tc.get(key)
        if v is not None:
            try:
                total += int(v)
            except (TypeError, ValueError):
                pass
    return total


def build_token_rollups(
    paragraphs_out: list[dict[str, Any]],
    *,
    system_prompt: str,
    user_prompt: str,
    num_vlm_calls: int,
    processor: Any | None = None,
    apply_chat_template: Callable[..., str] | None = None,
    num_images: int = 1,
    note: str | None = None,
) -> dict[str, Any]:
    sp_once, up_once, _, source = _count_sp_up(
        processor=processor,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        num_images=num_images,
        apply_chat_template=apply_chat_template,
        n_prompt=10**9,
    )
    if source == "qwen_tokenizer+chat_template":
        default_note = (
            "后端为 vLLM; SP/UP 为 Qwen tokenizer + chat_template 单请求计数 "
            "(与 vLLM prompt_token_ids 同分词器, 非 GPT tiktoken)。"
        )
    else:
        default_note = (
            "后端为 vLLM; SP/UP 为 tiktoken 回退估算(无 processor 时)。"
        )
    return {
        "num_api_calls": num_vlm_calls,
        "sum_input_prompt_tokens_total": sum_token_cost_field(
            paragraphs_out, "input_prompt_tokens_total",
        ),
        "sum_output_completion_tokens": sum_token_cost_field(
            paragraphs_out, "output_completion_tokens",
        ),
        "sum_image_tokens_fullpage": sum_token_cost_field(
            paragraphs_out, "image_tokens_fullpage",
        ),
        "sum_image_tokens_crop": sum_token_cost_field(
            paragraphs_out, "image_tokens_crop",
        ),
        "system_prompt_tokens_sp_per_request": sp_once,
        "user_prompt_text_tokens_up_per_request": up_once,
        "estimated_system_prompt_tokens_sp_per_request": sp_once,
        "estimated_user_prompt_text_tokens_up_per_request": up_once,
        "token_count_source": source,
        "_note": note or default_note,
    }


def vlm_prompts_record(system_prompt: str, user_prompt: str) -> dict[str, str]:
    return {"system_prompt": system_prompt, "user_prompt": user_prompt}


def enrich_profile_input_tokens(
    profile: dict[str, Any],
    *,
    processor: Any | None,
    system_prompt: str,
    user_prompt: str,
    global_context_wh: tuple[int, int],
    apply_chat_template: Callable[..., str] | None = None,
    num_images: int = 2,
    prefix_caching_enabled: bool = True,
) -> dict[str, Any]:
    """Augment the tokens section of __profile.json with the split between full-image prompt text tokens and visual tokens.

    Text = ``prompt_text_tokens_template`` x number of VLM calls;
    visual = vLLM ``total_prompt`` minus the text total (same accounting as
    ``build_vlm_token_cost``).
    """
    tokens = dict(profile.get("tokens") or {})
    num_vlm = int(profile.get("num_paragraphs") or 0)
    total_prompt = int(tokens.get("total_prompt") or 0)

    sp, up, text_tpl, source = _count_sp_up(
        processor=processor,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        num_images=num_images,
        apply_chat_template=apply_chat_template,
        n_prompt=total_prompt,
    )
    sum_text = text_tpl * num_vlm
    sum_vision = max(0, total_prompt - sum_text)
    sum_full, sum_crop = 0, 0

    per_para_out: list[dict[str, Any]] = []
    for row in profile.get("per_paragraph") or []:
        row = dict(row)
        pt = int(row.get("prompt_tokens") or 0)
        row["prompt_text_tokens"] = text_tpl
        row["prompt_vision_tokens"] = max(0, pt - text_tpl)
        cw = tuple(row.get("crop_wh") or [1, 1])
        gf, cf = split_image_tokens_by_pixels(
            row["prompt_vision_tokens"], global_context_wh, cw,
        )
        row["vision_tokens_fullpage"] = gf
        row["vision_tokens_crop"] = cf
        sum_full += gf
        sum_crop += cf
        per_para_out.append(row)

    total_completion = int(tokens.get("total_completion") or 0)
    template_overhead = max(0, text_tpl - sp - up)
    avg_vision_per_req = round(sum_vision / num_vlm, 1) if num_vlm else 0.0
    avg_decode_per_req = round(total_completion / num_vlm, 1) if num_vlm else 0.0

    per_request = {
        "system_prompt_tokens_sp": sp,
        "user_prompt_text_tokens_up": up,
        "chat_template_overhead_tokens": template_overhead,
        "prompt_text_tokens_template_total": text_tpl,
        "prompt_vision_tokens_avg_measured": avg_vision_per_req,
        "input_prompt_tokens_avg_measured": round(total_prompt / num_vlm, 1) if num_vlm else 0.0,
        "output_decode_tokens_avg_measured": avg_decode_per_req,
    }
    whole_image = {
        "num_vlm_calls": num_vlm,
        "sum_input_prompt_tokens": total_prompt,
        "sum_prompt_text_tokens": sum_text,
        "sum_prompt_vision_tokens": sum_vision,
        "sum_vision_tokens_fullpage": sum_full,
        "sum_vision_tokens_crop": sum_crop,
        "sum_output_decode_tokens": total_completion,
        "sum_io_tokens": total_prompt + total_completion,
    }

    tokens.update({
        "system_prompt_tokens_sp_per_request": sp,
        "user_prompt_text_tokens_up_per_request": up,
        "prompt_text_tokens_template_per_request": text_tpl,
        "chat_template_overhead_tokens_per_request": template_overhead,
        "sum_prompt_text_tokens": sum_text,
        "sum_prompt_vision_tokens": sum_vision,
        "sum_vision_tokens_fullpage": sum_full,
        "sum_vision_tokens_crop": sum_crop,
        "sum_output_decode_tokens": total_completion,
        "sum_io_tokens": total_prompt + total_completion,
        "text_pct": round(100.0 * sum_text / total_prompt, 1) if total_prompt else 0.0,
        "vision_pct": round(100.0 * sum_vision / total_prompt, 1) if total_prompt else 0.0,
        "token_count_source": source,
        "prefix_caching_enabled": prefix_caching_enabled,
        "per_request": per_request,
        "whole_image": whole_image,
    })
    profile = dict(profile)
    profile["tokens"] = tokens
    profile["vllm_prefix_caching"] = prefix_caching_enabled
    if per_para_out:
        profile["per_paragraph"] = per_para_out
    return profile


def print_token_breakdown(stem: str, profile: dict[str, Any]) -> None:
    """Print the SP/UP/image/decode breakdown using model-tokenizer counts + vLLM measurements."""
    tok = profile.get("tokens") or {}
    pr = tok.get("per_request") or {}
    wh = tok.get("whole_image") or {}
    source = tok.get("token_count_source", "?")
    n = int(wh.get("num_vlm_calls") or profile.get("num_paragraphs") or 0)

    print(f"\n{'=' * 72}")
    print(f"[{stem}] Token 明细  (文本: {source}; 输入/输出: vLLM prompt_token_ids)")
    print(f"{'=' * 72}")
    print("【单次 VLM 请求 · tokenizer 文本】")
    print(f"  SP (system):              {pr.get('system_prompt_tokens_sp', 0):>6}")
    print(f"  UP (user 文本):           {pr.get('user_prompt_text_tokens_up', 0):>6}")
    print(f"  chat 模板附加(占位符等):  {pr.get('chat_template_overhead_tokens', 0):>6}")
    print(f"  文本模板小计:             {pr.get('prompt_text_tokens_template_total', 0):>6}")
    print("【单次 VLM 请求 · vLLM 实测】")
    print(f"  输入 prompt (均值):       {pr.get('input_prompt_tokens_avg_measured', 0):>6}")
    print(f"    其中视觉(推):           {pr.get('prompt_vision_tokens_avg_measured', 0):>6}")
    print(f"  输出 decode (均值):       {pr.get('output_decode_tokens_avg_measured', 0):>6}")
    print("【整图累计 · vLLM 实测】")
    print(f"  输入 prompt 合计:         {wh.get('sum_input_prompt_tokens', 0):>8}")
    print(f"    文本模板 (×{n}):        {wh.get('sum_prompt_text_tokens', 0):>8}")
    print(f"    图片视觉合计:           {wh.get('sum_prompt_vision_tokens', 0):>8}")
    print(f"      · 整图上下文(估):     {wh.get('sum_vision_tokens_fullpage', 0):>8}")
    print(f"      · 段落 crop(估):      {wh.get('sum_vision_tokens_crop', 0):>8}")
    print(f"  输出 decode 合计:         {wh.get('sum_output_decode_tokens', 0):>8}")
    print(f"  输入+输出 IO 合计:        {wh.get('sum_io_tokens', 0):>8}")
    if tok.get("prefix_caching_enabled") is not None:
        print(f"  prefix_caching:           {tok.get('prefix_caching_enabled')}")

    per_para = profile.get("per_paragraph") or []
    if per_para:
        print("【各段落明细 · vLLM】")
        print(f"  {'pid':>5}  {'prompt':>7}  {'text':>6}  {'vision':>7}  "
              f"{'full':>5}  {'crop':>5}  {'decode':>7}")
        for row in per_para:
            pid = row.get("pid", "?")
            pt = int(row.get("prompt_tokens") or 0)
            dt = int(row.get("completion_tokens") or 0)
            txt = int(row.get("prompt_text_tokens") or 0)
            vis = int(row.get("prompt_vision_tokens") or 0)
            gf = int(row.get("vision_tokens_fullpage") or 0)
            cf = int(row.get("vision_tokens_crop") or 0)
            print(f"  {pid:>5}  {pt:>7}  {txt:>6}  {vis:>7}  {gf:>5}  {cf:>5}  {dt:>7}")
    print(f"{'=' * 72}\n")


def sum_nested_token_cost(
    paragraphs_out: list[dict[str, Any]],
    step_key: str,
    field: str,
) -> int:
    total = 0
    for p in paragraphs_out:
        tc = p.get("doubao_token_cost")
        if not isinstance(tc, dict):
            continue
        step_tc = tc.get(step_key)
        if not isinstance(step_tc, dict):
            continue
        v = step_tc.get(field)
        if v is not None:
            try:
                total += int(v)
            except (TypeError, ValueError):
                pass
    return total


def build_two_step_token_rollups(
    paragraphs_out: list[dict[str, Any]],
    *,
    step1_system_prompt: str,
    step1_user_prompt: str,
    step2_system_prompt: str,
    step2_user_prompt: str,
    num_step1_calls: int,
    num_step2_calls: int,
    processor: Any | None = None,
    apply_chat_template: Callable[..., str] | None = None,
) -> dict[str, Any]:
    sp1, up1, _, src1 = _count_sp_up(
        processor=processor,
        system_prompt=step1_system_prompt,
        user_prompt=step1_user_prompt,
        num_images=1,
        apply_chat_template=apply_chat_template,
        n_prompt=10**9,
    )
    sp2, up2, _, src2 = _count_sp_up(
        processor=processor,
        system_prompt=step2_system_prompt,
        user_prompt=step2_user_prompt,
        num_images=1,
        apply_chat_template=apply_chat_template,
        n_prompt=10**9,
    )
    source = src1 if src1 == src2 else f"{src1}|{src2}"
    return {
        "num_api_calls_step1_translation": num_step1_calls,
        "num_api_calls_step2_style": num_step2_calls,
        "step1_translation": {
            "sum_input_prompt_tokens_total": sum_nested_token_cost(
                paragraphs_out, "step1_translation", "input_prompt_tokens_total",
            ),
            "sum_output_completion_tokens": sum_nested_token_cost(
                paragraphs_out, "step1_translation", "output_completion_tokens",
            ),
            "system_prompt_tokens_sp_per_request": sp1,
            "user_prompt_text_tokens_up_per_request": up1,
            "estimated_system_prompt_tokens_sp_per_request": sp1,
            "estimated_user_prompt_text_tokens_up_per_request": up1,
        },
        "step2_style": {
            "sum_input_prompt_tokens_total": sum_nested_token_cost(
                paragraphs_out, "step2_style", "input_prompt_tokens_total",
            ),
            "sum_output_completion_tokens": sum_nested_token_cost(
                paragraphs_out, "step2_style", "output_completion_tokens",
            ),
            "system_prompt_tokens_sp_per_request": sp2,
            "user_prompt_text_tokens_up_per_request": up2,
            "estimated_system_prompt_tokens_sp_per_request": sp2,
            "estimated_user_prompt_text_tokens_up_per_request": up2,
        },
        "token_count_source": source,
        "_note": (
            "两步 vLLM: step1 译文字段 / step2 样式字段; "
            "SP/UP 为 Qwen tokenizer + chat_template 单请求计数。"
        ),
    }
