#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

"""
Batch image translation (local open-source models). Automatically detects whether
the weights referenced by ``--model-path`` are a VLM or a text-only LM:
  - **VLM** (e.g. Qwen3-VL-8B-Instruct, Qwen3.5-9B in this directory): end-to-end
    image-to-translation.
  - **Text-only LM**: two stages -- the VL pointed to by ``--vision-model-path`` first
    extracts the original text from the image, then the text LM (``--model-path``)
    translates it into Chinese.

The output layout matches a provider-specific batch translator:
  <output-root>/<sanitized-model-name>/
    translations/<stem>.txt
    per_image_json/<stem>.json
    summary.json

Usage:
  # End-to-end VL (Transformers)
  python batch_photo_translate_vlm.py \
      --model-path /path/to/models/Qwen3-VL-8B-Instruct

  # End-to-end VL (vLLM; requires CUDA; throughput is usually much better than HF generate)
  python batch_photo_translate_vlm.py \
      --model-path /path/to/models/Qwen3-VL-8B-Instruct \
      --backend vllm

  # vLLM: when the first inference is slow, warm up first (no output is written),
  # then process input-dir
  python batch_photo_translate_vlm.py --backend vllm ... \
      --vllm-warmup-image /path/to/any_other.jpg \
      --input-dir /path/to/target_images_only

  # Qwen3.5-VL (example directory name Qwen3.5-9B) is also end-to-end + vLLM
  python batch_photo_translate_vlm.py \
      --model-path /path/to/models/Qwen3.5-9B \
      --backend vllm

  # Two stages (text-only LM + a separate VL) -- by default the vision side reuses
  # the local Qwen3-VL-8B
  python batch_photo_translate_vlm.py \
      --model-path /path/to/text-only-LM \
      --pipeline two-stage

  # Force a specific pipeline
  python batch_photo_translate_vlm.py --model-path ... --pipeline two-stage

HGGC / PPU (e.g. PPU-ZW810E) with ``--backend vllm``:
  - **Recommended**: first run ``source ../image-translation/setup_vllm_ppu_env.sh``
    (sets LD_LIBRARY_PATH, ``PPU_SDK``, CUDA, TORCH_COMPILE_DISABLE, etc.).
  - If not sourced, this script will **best-effort** fill in ``PPU_SDK`` / PATH / common
    environment variables (compatible with subprocesses) when it detects the HGGC SDK;
    for ``libhggcrt`` it is still recommended to set ``LD_LIBRARY_PATH`` **before
    starting the interpreter**.
  - To force-enable Inductor on PPU anyway (not recommended):
    ``export VLLM_FORCE_INDUCTOR=1``.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


IMAGE_TRANSLATE_PROMPT = """请根据这张图片完成「翻译」类输出（目标语言：中文）：
- 若图中有可阅读的文字（任意语种），请准确识别并按其在图中的阅读顺序与层次，全部翻译成流畅、自然的中文。翻译时必须结合**整图语境**（场景、主题、符号与相邻条目、标题与正文关系等），保证**语义连贯、指代一致**，避免只看局部词面而造成生硬或歧义；专名、品牌、数字与单位在合理情况下可保留或附中文说明，勿臆造。
- **严禁复读与填充**：不得重复输出同一段译文、同一条列表项或同一表述；不得在文末用重复的数字、标点或符号拉长输出；成分表、剂量表、警告条款等只需按图中实际出现的内容译**一遍**，概括完整即可，**禁止**循环抄写相同条目或整段复述。
- 若几乎没有可读文字，也不要编造图中不存在的内容。
- 只输出最终中文结果，不要思考过程、不要道歉、不要评价任务本身；信息量译完即止，不要为了凑字数继续生成。"""


EXTRACT_OCR_PROMPT = """请**只输出**图中所有可读文字的**原文摘录**：
- 按照图中的阅读顺序与版面分块关系列出，可用简单列表或分段；
- 保留原语种，不要翻译成中文，不要解释，不要添加任何评论；
- 若同一张图中出现多种语言，分别保留它们的原文；
- 若图中几乎无可读文字或无法辨认，则仅输出：图中无可摘录文字。"""


TEXT_TRANSLATE_PROMPT_TMPL = (
    "下面是从**同一张**照片中识别出的文字原文（可能含多种语言），请全部译为流畅、自然的中文。"
    "请把整段文字视为同一画面下的内容：结合**前后文与主题**统一选词与语气，保持**语义连贯、指代一致**，"
    "不要孤立硬译；保持原有顺序与层次对应。**只输出最终中文译文，不要重复原文、不要解释、不要思考过程；"
    "禁止循环重复同一段列表或条目（如成分表），每条信息只译一次。**"
    "\n\n<<<SOURCE>>>\n{source}\n<<<END>>>"
)


_THINK_OPEN_RE = re.compile(r"<think\s*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Strip the <think>...</think> section (including missing close tags or multiple sections) and return the actual answer body."""
    if not text:
        return text
    m_close = list(_THINK_CLOSE_RE.finditer(text))
    if m_close:
        return text[m_close[-1].end():].lstrip()
    m_open = _THINK_OPEN_RE.search(text)
    if m_open:
        head = text[: m_open.start()].strip()
        return head or text.strip()
    return text.strip()


def _safe_dirname(s: str) -> str:
    s = re.sub(r"[^\w.\-]+", "_", s.strip())
    return s or "model"


def list_image_paths(directory: Path, limit: int, extensions: tuple[str, ...]) -> list[str]:
    paths: list[str] = []
    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in extensions:
            continue
        paths.append(str(p.resolve()))
        if len(paths) >= limit:
            break
    return paths


def detect_model_kind(model_path: str) -> tuple[str, str, str]:
    """Inspect config.json to decide 'vl' | 'text'; returns (kind, architecture, model_type).

    Criteria (strongest first):
      1) config has a vision_config sub-block -> VL
      2) config has multimodal tokens such as image_token_id / vision_start_token_id -> VL
      3) the directory contains preprocessor_config.json (image processor) -> VL
      4) otherwise treat as a text-only LM
    This avoids hard-coded guessing from the architecture name (the Qwen3.5-VL
    architecture is called Qwen3_5ForConditionalGeneration; the name contains neither
    "VL" nor Qwen3VL, but it is in fact a VLM).
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    archs = getattr(cfg, "architectures", None) or []
    arch = archs[0] if archs else ""
    model_type = getattr(cfg, "model_type", "") or ""

    cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else {}
    has_vision_cfg = bool(cfg_dict.get("vision_config"))
    has_image_token = any(
        cfg_dict.get(k) is not None
        for k in ("image_token_id", "vision_start_token_id", "vision_end_token_id", "video_token_id")
    )

    has_image_processor = False
    try:
        for name in ("preprocessor_config.json", "image_processor.json"):
            if (Path(model_path) / name).exists():
                has_image_processor = True
                break
    except Exception:
        pass

    if has_vision_cfg or has_image_token or has_image_processor:
        return "vl", arch, model_type
    return "text", arch, model_type


def _resolve_dtype(dtype_str: str):
    import torch

    return {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(dtype_str, "auto")


def _load_vl_model(
    model_path: str,
    *,
    dtype: str,
    attn_implementation: str | None,
    device_map: str,
):
    """Load VLM weights.

    Order:
      1) Read config.architectures[0]; if transformers happens to have a class with
         the same name, use it directly (this is the most **authoritative** approach;
         Qwen3.5-VL's Qwen3_5ForConditionalGeneration relies on it)
      2) Common VL fallbacks: Qwen3VL / Qwen2_5_VL
      3) Generic fallback: AutoModelForImageTextToText
    """
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor

    kwargs: dict[str, Any] = {"device_map": device_map, "trust_remote_code": True}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    resolved = _resolve_dtype(dtype)

    def _from_pretrained(cls):
        try:
            return cls.from_pretrained(model_path, dtype=resolved, **kwargs)
        except TypeError:
            td = resolved if resolved != "auto" else torch.bfloat16
            return cls.from_pretrained(model_path, torch_dtype=td, **kwargs)

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    arch_list = list(getattr(cfg, "architectures", None) or [])

    tried: list[str] = []
    last_err: Exception | None = None

    for cls_name in arch_list:
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        tried.append(cls_name)
        try:
            model = _from_pretrained(cls)
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            return model, processor, cls_name
        except Exception as e:
            last_err = e

    for cls_name in (
        "Gemma4ForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
    ):
        cls = getattr(transformers, cls_name, None)
        if cls is None or cls_name in tried:
            continue
        tried.append(cls_name)
        try:
            model = _from_pretrained(cls)
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            return model, processor, cls_name
        except Exception as e:
            last_err = e

    try:
        from transformers import AutoModelForImageTextToText

        tried.append("AutoModelForImageTextToText")
        model = _from_pretrained(AutoModelForImageTextToText)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return model, processor, "AutoModelForImageTextToText"
    except Exception as e:
        last_err = e

    raise RuntimeError(
        f"无法将 {model_path} 作为 VL 模型加载；已尝试 {tried}，最后错误: {last_err!r}\n"
        f"请升级 transformers：pip install -U 'transformers>=4.47'"
    )


def _load_text_model(
    model_path: str,
    *,
    dtype: str,
    attn_implementation: str | None,
    device_map: str,
):
    """Load a text-only LM (AutoModelForCausalLM)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict[str, Any] = {"device_map": device_map, "trust_remote_code": True}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    resolved = _resolve_dtype(dtype)

    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=resolved, **kwargs)
    except TypeError:
        td = resolved if resolved != "auto" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=td, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, trust_remote_code=True
    )
    return model, tokenizer, "AutoModelForCausalLM"


def _apply_chat_template_no_think(proc_or_tok, messages, **kwargs):
    """Prefer disabling Qwen3's thinking mode; degrade gracefully if the underlying template does not support that parameter."""
    try:
        return proc_or_tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return proc_or_tok.apply_chat_template(messages, **kwargs)


def _vl_generate(
    *,
    model,
    processor,
    image_path: str,
    user_text: str,
    max_new_tokens: int,
) -> tuple[str, dict[str, int], float]:
    """Single image + text -> (translation/extraction, usage, elapsed seconds); <think> is stripped automatically."""
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    t0 = time.perf_counter()
    inputs = _apply_chat_template_no_think(
        processor,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    prompt_len = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.12,
        )
    elapsed = time.perf_counter() - t0
    gen_trim = generated_ids[:, prompt_len:]
    raw = processor.batch_decode(
        gen_trim,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    text = _strip_thinking(raw.strip())
    completion_tokens = int(gen_trim.shape[1])
    usage = {
        "prompt_tokens": prompt_len,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_len + completion_tokens,
    }
    return text, usage, elapsed


def _text_generate(
    *,
    model,
    tokenizer,
    user_text: str,
    max_new_tokens: int,
) -> tuple[str, dict[str, int], float]:
    """Text-only LM generation -> (text, usage, elapsed seconds); <think> is stripped automatically."""
    import torch

    messages = [{"role": "user", "content": user_text}]
    t0 = time.perf_counter()
    prompt = _apply_chat_template_no_think(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.12,
        )
    elapsed = time.perf_counter() - t0
    gen_trim = out_ids[:, prompt_len:]
    raw = tokenizer.decode(gen_trim[0], skip_special_tokens=True).strip()
    text = _strip_thinking(raw)
    completion_tokens = int(gen_trim.shape[1])
    usage = {
        "prompt_tokens": prompt_len,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_len + completion_tokens,
    }
    return text, usage, elapsed


def _warn_if_vllm_without_cuda() -> None:
    """Give a gentle hint about whether a GPU is present.

    **Do not** call ``torch.cuda.is_available()`` here: that API initializes CUDA in
    the main process, which later triggers
    ``Cannot re-initialize CUDA in forked subprocess`` when vLLM spawns EngineCore via fork.
    """
    import shutil
    from pathlib import Path

    if Path("/dev/nvidia0").exists():
        return
    if shutil.which("nvidia-smi"):
        return
    print(
        "[WARN] --backend vllm：未检测到常见 NVIDIA 设备线索（无 /dev/nvidia0、无 nvidia-smi）；"
        "若无 CUDA GPU 请改用 --backend transformers",
        file=sys.stderr,
    )


def _ensure_vllm_worker_multiproc_spawn() -> None:
    """When using vLLM as a library this must be set before creating the LLM (consistent with vllm.entrypoints.utils.cli_env_setup)."""
    if "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _vllm_dtype_literal(dtype: str) -> str | None:
    """Map --dtype to the vLLM LLM(dtype=...) literal; None means do not pass it (auto)."""
    d = (dtype or "auto").strip().lower()
    return {
        "auto": None,
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }.get(d)


def _vllm_hf_overrides_fix_text_config(model_path: str) -> dict[str, Any] | None:
    """Fix the ``text_config`` passed to vLLM ``LLM(hf_overrides=...)`` to avoid conflicts with the current vLLM+HF behavior.

    1) **tie_word_embeddings**: commonly present at the root but missing in
       ``text_config``; when vLLM only reads the sub-config it raises
       ``AttributeError``, so merge the root-level boolean into it.

    2) **qwen3_5 + dtype string**: the official ``config.json`` often writes
       ``"dtype": "bfloat16"`` inside ``text_config``. After loading,
       ``Qwen3_5TextConfig.dtype`` is still a **str**, while
       ``Qwen3_5GatedDeltaNet`` in ``vllm/model_executor/models/qwen3_5.py`` passes
       ``config.dtype`` straight to ``torch.empty(..., dtype=..., device=...)``,
       triggering ``TypeError: ... dtype=str, device=int``. Removing this key lets
       ``LLM(dtype=...)`` / the weights infer it.
    """
    p = Path(model_path) / "config.json"
    if not p.is_file():
        return None
    try:
        raw: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    tc = raw.get("text_config")
    if not isinstance(tc, dict):
        return None
    new_tc: dict[str, Any] = dict(tc)
    changed = False

    if "tie_word_embeddings" not in new_tc and "tie_word_embeddings" in raw:
        t = raw.get("tie_word_embeddings")
        new_tc["tie_word_embeddings"] = False if t is None else t
        changed = True

    if raw.get("model_type") == "qwen3_5" and isinstance(new_tc.get("dtype"), str):
        new_tc.pop("dtype", None)
        changed = True

    if not changed:
        return None
    return {"text_config": new_tc}


def _hggc_ppu_sdk_root() -> str | None:
    """If an HGGC runtime library exists on this machine, return the PPU_SDK root path; otherwise None."""
    for lib in (
        "/usr/local/PPU_SDK/lib/libhggcrt1.so",
        "/usr/local/PPU_SDK/lib/libhggcrt.12.0.so",
        "/opt/PPU_SDK/lib/libhggcrt1.so",
        "/opt/PPU_SDK/lib/libhggcrt.12.0.so",
    ):
        if os.path.isfile(lib):
            return str(Path(lib).resolve().parents[1])
    return None


def _ensure_hggc_runtime_env_for_vllm() -> None:
    """Best-effort fill of common HGGC variables before EngineCore subprocess/Triton compilation (aligned with image-translation/setup_vllm_ppu_env.sh).

    When ``PPU_SDK`` is unset, Triton (e.g. mrope) raises ``TypeError`` inside the
    subprocess; here the root directory is inferred from the installed ``libhggcrt``.
    """
    root = _hggc_ppu_sdk_root()
    if not root:
        return
    if not os.environ.get("PPU_SDK"):
        os.environ["PPU_SDK"] = root
    os.environ.setdefault("VLLM_DISABLE_INDUCTOR", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
    lib_dir = os.path.join(root, "lib")
    if os.path.isdir(lib_dir):
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [x for x in cur.split(os.pathsep) if x]
        if lib_dir not in parts:
            os.environ["LD_LIBRARY_PATH"] = (
                lib_dir + (os.pathsep + cur if cur else "")
            )
    bin_dir = os.path.join(root, "bin")
    if os.path.isdir(bin_dir):
        path = os.environ.get("PATH", "")
        pparts = [x for x in path.split(os.pathsep) if x]
        if bin_dir not in pparts:
            os.environ["PATH"] = bin_dir + (os.pathsep + path if path else "")


def _vllm_should_disable_inductor(enforce_eager: bool) -> bool:
    """On HGGC/PPU-like environments Torch Inductor often fails during codegen; explicitly disable vLLM_COMPILE.

    - ``export VLLM_DISABLE_INDUCTOR=1``: force-disable (even without ``--enforce-eager``).
    - Disabled by default when an HGGC SDK directory is detected; to experimentally
      force-enable: ``export VLLM_FORCE_INDUCTOR=1``.
    """
    if os.environ.get("VLLM_FORCE_INDUCTOR", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    if enforce_eager:
        return True
    v = os.environ.get("VLLM_DISABLE_INDUCTOR", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return _hggc_ppu_sdk_root() is not None


def _vllm_llm_kwargs_disable_inductor_stack() -> dict[str, Any]:
    """Disable the entire torch.compile/Inductor stack (including the extras vLLM O2 applies on top of compilation_config by default).

    ``compilation_config=0`` alone is not enough: when ``optimization_level`` stays at
    O2 the defaults such as piecewise/cudagraph are still merged in; in addition,
    several internal vLLM ``@torch.compile(backend=simple_compile_backend)`` sites also
    use inductor by default, so the environment variable ``TORCH_COMPILE_DISABLE=1`` is
    also required (see ``setup_vllm_ppu_env.sh``).
    """
    from vllm.config.compilation import CUDAGraphMode
    from vllm.config.vllm import OptimizationLevel

    return {
        "optimization_level": OptimizationLevel.O0,
        "compilation_config": {
            "mode": 0,
            "backend": "eager",
            "cudagraph_mode": CUDAGraphMode.NONE,
        },
    }


def _load_vllm_vl_engine(
    model_path: str,
    *,
    dtype: str,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    max_model_len: int | None,
    enforce_eager: bool,
    extra_engine_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, Any, str]:
    """Load a multimodal VLM with vLLM; only the HF Processor is used for the chat template.

    ``extra_engine_kwargs``: additional ``LLM(**kw)`` parameters supplied by the caller
    (e.g. ``mm_processor_kwargs``, ``enable_prefix_caching``, ``max_num_seqs``).
    They are ``update``d after the built-in kwargs and may override defaults such as
    ``limit_mm_per_prompt``.
    """
    # step1_ocr_paragraph_vlm*.py and similar callers invoke this function directly
    # without going through ``main()``, so the environment must be patched here;
    # otherwise the EngineCore subprocess cannot find libhggcrt / PPU_SDK is unset.
    _ensure_hggc_runtime_env_for_vllm()
    from transformers import AutoProcessor
    from vllm import LLM

    _ensure_vllm_worker_multiproc_spawn()
    kw: dict[str, Any] = {
        "model": model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "limit_mm_per_prompt": {"image": 1},
        "enforce_eager": enforce_eager,
    }
    if _vllm_should_disable_inductor(enforce_eager):
        kw.update(_vllm_llm_kwargs_disable_inductor_stack())
    dt = _vllm_dtype_literal(dtype)
    if dt:
        kw["dtype"] = dt
    if max_model_len is not None and max_model_len > 0:
        kw["max_model_len"] = max_model_len
    ov = _vllm_hf_overrides_fix_text_config(model_path)
    if ov:
        kw["hf_overrides"] = ov
    if extra_engine_kwargs:
        kw.update({k: v for k, v in extra_engine_kwargs.items() if v is not None})
    llm = LLM(**kw)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return llm, processor, "vLLM"


def _load_vllm_text_engine(
    model_path: str,
    *,
    dtype: str,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    max_model_len: int | None,
    enforce_eager: bool,
) -> tuple[Any, Any, str]:
    """Load a text-only LM with vLLM (second step of the two-stage pipeline)."""
    _ensure_hggc_runtime_env_for_vllm()
    from transformers import AutoTokenizer
    from vllm import LLM

    _ensure_vllm_worker_multiproc_spawn()
    kw: dict[str, Any] = {
        "model": model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": enforce_eager,
    }
    if _vllm_should_disable_inductor(enforce_eager):
        kw.update(_vllm_llm_kwargs_disable_inductor_stack())
    dt = _vllm_dtype_literal(dtype)
    if dt:
        kw["dtype"] = dt
    if max_model_len is not None and max_model_len > 0:
        kw["max_model_len"] = max_model_len
    llm = LLM(**kw)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, trust_remote_code=True
    )
    return llm, tokenizer, "vLLM"


def _vl_generate_vllm(
    *,
    llm,
    processor,
    image_path: str,
    user_text: str,
    max_new_tokens: int,
) -> tuple[str, dict[str, int], float]:
    """Single image + text -> (translation/extraction, usage, elapsed); aligned with _vl_generate but goes through vLLM."""
    from PIL import Image
    from vllm import SamplingParams

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    prompt = _apply_chat_template_no_think(
        processor,
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    pil = Image.open(image_path).convert("RGB")
    sp = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        repetition_penalty=1.12,
    )
    t0 = time.perf_counter()
    outs = llm.generate(
        [{"prompt": prompt, "multi_modal_data": {"image": pil}}],
        sampling_params=sp,
        use_tqdm=False,
    )
    elapsed = time.perf_counter() - t0
    ro = outs[0]
    comp = ro.outputs[0]
    raw = (comp.text or "").strip()
    text = _strip_thinking(raw)
    prompt_ids = ro.prompt_token_ids or []
    prompt_len = len(prompt_ids)
    completion_len = len(comp.token_ids)
    usage = {
        "prompt_tokens": prompt_len,
        "completion_tokens": completion_len,
        "total_tokens": prompt_len + completion_len,
    }
    return text, usage, elapsed


def _text_generate_vllm(
    *,
    llm,
    tokenizer,
    user_text: str,
    max_new_tokens: int,
) -> tuple[str, dict[str, int], float]:
    """Text-only LM (vLLM)."""
    from vllm import SamplingParams

    messages = [{"role": "user", "content": user_text}]
    prompt = _apply_chat_template_no_think(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    sp = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        repetition_penalty=1.12,
    )
    t0 = time.perf_counter()
    outs = llm.generate([prompt], sampling_params=sp, use_tqdm=False)
    elapsed = time.perf_counter() - t0
    ro = outs[0]
    comp = ro.outputs[0]
    raw = (comp.text or "").strip()
    text = _strip_thinking(raw)
    prompt_len = len(ro.prompt_token_ids or [])
    completion_len = len(comp.token_ids)
    usage = {
        "prompt_tokens": prompt_len,
        "completion_tokens": completion_len,
        "total_tokens": prompt_len + completion_len,
    }
    return text, usage, elapsed


def _write_per_image(
    *,
    meta_dir: Path,
    translations_dir: Path,
    stem: str,
    rec: dict[str, Any],
) -> tuple[Path, Path | None]:
    per_path = meta_dir / f"{stem}.json"
    with open(per_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    trans_path: Path | None = None
    if rec.get("translation"):
        trans_path = translations_dir / f"{stem}.txt"
        with open(trans_path, "w", encoding="utf-8") as f:
            f.write(rec["translation"])
    return per_path, trans_path


def _resume_read(per_path: Path) -> dict[str, Any] | None:
    if not per_path.exists():
        return None
    try:
        with open(per_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _maybe_vllm_warmup_vl_one_shot(
    *,
    backend: str,
    llm: Any,
    processor: Any,
    warmup_image: str | None,
    warmup_max_new_tokens: int,
    user_prompt: str,
) -> None:
    """The first vLLM image often carries first-round JIT/caching overhead; optionally run generate once on an arbitrary image, without counting it in the output."""
    if backend != "vllm" or llm is None or not (warmup_image or "").strip():
        return
    p = Path(warmup_image.strip()).expanduser().resolve()
    if not p.is_file():
        print(f"[WARN] --vllm-warmup-image 不存在，跳过预热: {p}", file=sys.stderr)
        return
    print(f"[vLLM warmup] {p.name}（不计入输出）…")
    wall0 = time.perf_counter()
    try:
        _usage: dict[str, Any]
        _, _usage, elapsed = _vl_generate_vllm(
            llm=llm,
            processor=processor,
            image_path=str(p),
            user_text=user_prompt,
            max_new_tokens=max(16, int(warmup_max_new_tokens)),
        )
        print(
            f"[vLLM warmup] 完成 tokens={_usage.get('total_tokens')} "
            f"infer={elapsed:.3f}s wall={time.perf_counter() - wall0:.3f}s"
        )
    except Exception as e:
        print(f"[WARN] vLLM 预热失败（继续正式推理）: {e!r}", file=sys.stderr)


def run_end2end_vl(
    *,
    model_path: str,
    model_name: str,
    image_paths: list[str],
    out_dir: Path,
    max_new_tokens: int,
    dtype: str,
    device_map: str,
    attn_implementation: str | None,
    resume: bool,
    architecture: str,
    model_type: str,
    backend: str,
    vllm_gpu_memory_utilization: float,
    vllm_tensor_parallel_size: int,
    vllm_max_model_len: int | None,
    vllm_enforce_eager: bool,
    vllm_warmup_image: str | None,
    vllm_warmup_max_new_tokens: int,
) -> None:
    translations_dir = out_dir / "translations"
    meta_dir = out_dir / "per_image_json"
    translations_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    hf_model = None
    vl_llm = None
    processor = None
    cls_name = ""

    if backend == "vllm":
        print(
            f"[VL end-to-end] 加载 VLM (vLLM): {model_path}\n"
            f"  dtype={dtype} tensor_parallel={vllm_tensor_parallel_size} "
            f"gpu_mem_util={vllm_gpu_memory_utilization} max_model_len={vllm_max_model_len}"
        )
        t_load = time.perf_counter()
        vl_llm, processor, cls_name = _load_vllm_vl_engine(
            model_path,
            dtype=dtype,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            tensor_parallel_size=vllm_tensor_parallel_size,
            max_model_len=vllm_max_model_len,
            enforce_eager=vllm_enforce_eager,
        )
    else:
        print(
            f"[VL end-to-end] 加载 VLM (transformers): {model_path}\n"
            f"  dtype={dtype} device_map={device_map} attn={attn_implementation}"
        )
        t_load = time.perf_counter()
        hf_model, processor, cls_name = _load_vl_model(
            model_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )
    print(f"  模型就绪 ({cls_name})，加载耗时 {time.perf_counter() - t_load:.2f}s")

    _maybe_vllm_warmup_vl_one_shot(
        backend=backend,
        llm=vl_llm,
        processor=processor,
        warmup_image=vllm_warmup_image,
        warmup_max_new_tokens=vllm_warmup_max_new_tokens,
        user_prompt=IMAGE_TRANSLATE_PROMPT,
    )

    summary: list[dict[str, Any]] = []
    try:
        for i, img_path in enumerate(image_paths):
            stem = Path(img_path).stem
            per_path = meta_dir / f"{stem}.json"
            if resume:
                prev = _resume_read(per_path)
                if prev is not None and prev.get("translation"):
                    summary.append(_summary_item(img_path, per_path, translations_dir, stem, prev))
                    print(f"[{i + 1}/{len(image_paths)}] {stem} skip (resume)")
                    continue

            rec: dict[str, Any] = {
                "index": i,
                "image_path": img_path,
                "pipeline": "vl_end2end",
                "backend": backend,
                "model": model_name,
                "model_path": model_path,
                "model_class": cls_name,
                "architecture": architecture,
                "model_type": model_type,
                "error": None,
                "translation": None,
                "usage": {},
                "translation_time_seconds": None,
            }
            try:
                if backend == "vllm":
                    text, usage, elapsed = _vl_generate_vllm(
                        llm=vl_llm,
                        processor=processor,
                        image_path=img_path,
                        user_text=IMAGE_TRANSLATE_PROMPT,
                        max_new_tokens=max_new_tokens,
                    )
                else:
                    text, usage, elapsed = _vl_generate(
                        model=hf_model,
                        processor=processor,
                        image_path=img_path,
                        user_text=IMAGE_TRANSLATE_PROMPT,
                        max_new_tokens=max_new_tokens,
                    )
                rec["translation"] = text
                rec["usage"] = usage
                rec["translation_time_seconds"] = round(elapsed, 6)
            except Exception as e:
                rec["error"] = repr(e)

            _write_per_image(
                meta_dir=meta_dir,
                translations_dir=translations_dir,
                stem=stem,
                rec=rec,
            )
            summary.append(_summary_item(img_path, per_path, translations_dir, stem, rec))
            print(
                f"[{i + 1}/{len(image_paths)}] {stem} "
                f"tokens={rec['usage'].get('total_tokens')} "
                f"time={rec['translation_time_seconds']}s "
                f"err={'yes' if rec['error'] else 'no'}"
            )
    finally:
        if backend == "vllm":
            _cleanup(vl_llm, processor)
        else:
            _cleanup(hf_model, processor)

    sum_base: dict[str, Any] = {
        "pipeline": "vl_end2end",
        "backend": backend,
        "model": model_name,
        "model_path": model_path,
        "model_class": cls_name,
        "architecture": architecture,
        "model_type": model_type,
        "count": len(image_paths),
        "max_new_tokens": max_new_tokens,
        "dtype": dtype,
        "prompt": IMAGE_TRANSLATE_PROMPT,
        "items": summary,
    }
    if backend == "vllm":
        sum_base["vllm"] = {
            "tensor_parallel_size": vllm_tensor_parallel_size,
            "gpu_memory_utilization": vllm_gpu_memory_utilization,
            "max_model_len": vllm_max_model_len,
            "enforce_eager": vllm_enforce_eager,
            "warmup_image": vllm_warmup_image,
            "warmup_max_new_tokens": vllm_warmup_max_new_tokens,
        }
    else:
        sum_base["device_map"] = device_map
        sum_base["attn_implementation"] = attn_implementation

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(sum_base, f, ensure_ascii=False, indent=2)


def run_two_stage(
    *,
    vision_model_path: str,
    text_model_path: str,
    model_name: str,
    image_paths: list[str],
    out_dir: Path,
    max_new_tokens_vl: int,
    max_new_tokens_text: int,
    dtype: str,
    device_map: str,
    attn_implementation: str | None,
    unload_vision_between_stages: bool,
    resume: bool,
    text_architecture: str,
    text_model_type: str,
    backend: str,
    vllm_gpu_memory_utilization: float,
    vllm_tensor_parallel_size: int,
    vllm_max_model_len: int | None,
    vllm_enforce_eager: bool,
    vllm_warmup_image: str | None,
    vllm_warmup_max_new_tokens: int,
) -> None:
    translations_dir = out_dir / "translations"
    meta_dir = out_dir / "per_image_json"
    translations_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    vl_hf = None
    vl_llm = None
    vl_processor = None
    vl_cls = ""

    head = (
        f"[two-stage] VL 抽原文: {vision_model_path}\n"
        f"           文本翻译: {text_model_path}\n"
        f"  backend={backend} dtype={dtype}"
    )
    if backend == "vllm":
        head += (
            f" tensor_parallel={vllm_tensor_parallel_size} "
            f"gpu_mem_util={vllm_gpu_memory_utilization}"
        )
    else:
        head += f" device_map={device_map} attn={attn_implementation}"
    print(head)
    if backend == "vllm" and not unload_vision_between_stages:
        print(
            "[WARN] 两阶段且 --backend vllm：未指定 --unload-vision-between-stages 时会"
            "同时在显存里驻留 VL 与文本两套 vLLM 引擎，显存不足请加该选项分阶段加载。",
            file=sys.stderr,
        )

    t_load = time.perf_counter()
    if backend == "vllm":
        vl_llm, vl_processor, vl_cls = _load_vllm_vl_engine(
            vision_model_path,
            dtype=dtype,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            tensor_parallel_size=vllm_tensor_parallel_size,
            max_model_len=vllm_max_model_len,
            enforce_eager=vllm_enforce_eager,
        )
    else:
        vl_hf, vl_processor, vl_cls = _load_vl_model(
            vision_model_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )
    print(f"  VL 就绪 ({vl_cls})，加载耗时 {time.perf_counter() - t_load:.2f}s")

    _maybe_vllm_warmup_vl_one_shot(
        backend=backend,
        llm=vl_llm,
        processor=vl_processor,
        warmup_image=vllm_warmup_image,
        warmup_max_new_tokens=vllm_warmup_max_new_tokens,
        user_prompt=EXTRACT_OCR_PROMPT,
    )

    extractions: list[dict[str, Any]] = []
    try:
        for i, img_path in enumerate(image_paths):
            stem = Path(img_path).stem
            per_path = meta_dir / f"{stem}.json"
            row: dict[str, Any] = {
                "stem": stem,
                "image_path": img_path,
                "source_text": None,
                "extract_usage": None,
                "extract_time_seconds": None,
                "extract_error": None,
            }
            prev = _resume_read(per_path) if resume else None
            if prev is not None and prev.get("extracted_source_text"):
                row["source_text"] = prev["extracted_source_text"]
                row["extract_usage"] = (prev.get("stages") or [{}])[0].get("usage")
                row["extract_time_seconds"] = (prev.get("stages") or [{}])[0].get(
                    "time_seconds"
                )
                extractions.append(row)
                print(f"[extract {i + 1}/{len(image_paths)}] {stem} skip (resume)")
                continue

            try:
                if backend == "vllm":
                    src, usage, elapsed = _vl_generate_vllm(
                        llm=vl_llm,
                        processor=vl_processor,
                        image_path=img_path,
                        user_text=EXTRACT_OCR_PROMPT,
                        max_new_tokens=max_new_tokens_vl,
                    )
                else:
                    src, usage, elapsed = _vl_generate(
                        model=vl_hf,
                        processor=vl_processor,
                        image_path=img_path,
                        user_text=EXTRACT_OCR_PROMPT,
                        max_new_tokens=max_new_tokens_vl,
                    )
                row["source_text"] = src
                row["extract_usage"] = usage
                row["extract_time_seconds"] = round(elapsed, 6)
            except Exception as e:
                row["extract_error"] = repr(e)
            extractions.append(row)
            print(
                f"[extract {i + 1}/{len(image_paths)}] {stem} "
                f"err={'yes' if row['extract_error'] else 'no'}"
            )
    finally:
        if unload_vision_between_stages:
            if backend == "vllm":
                _cleanup(vl_llm, vl_processor)
                vl_llm = None
                vl_processor = None
            else:
                _cleanup(vl_hf, vl_processor)
                vl_hf = None
                vl_processor = None

    print("  加载文本 LM …")
    text_hf = None
    text_llm = None
    tokenizer = None
    text_cls = ""
    t_load = time.perf_counter()
    if backend == "vllm":
        text_llm, tokenizer, text_cls = _load_vllm_text_engine(
            text_model_path,
            dtype=dtype,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            tensor_parallel_size=vllm_tensor_parallel_size,
            max_model_len=vllm_max_model_len,
            enforce_eager=vllm_enforce_eager,
        )
    else:
        text_hf, tokenizer, text_cls = _load_text_model(
            text_model_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )
    print(f"  文本 LM 就绪 ({text_cls})，加载耗时 {time.perf_counter() - t_load:.2f}s")

    summary: list[dict[str, Any]] = []
    try:
        for i, ext in enumerate(extractions):
            stem = ext["stem"]
            per_path = meta_dir / f"{stem}.json"
            rec: dict[str, Any] = {
                "index": i,
                "image_path": ext["image_path"],
                "pipeline": "two_stage_vl_extract_then_text_translate",
                "backend": backend,
                "model": model_name,
                "models": {
                    "vision": vision_model_path,
                    "text": text_model_path,
                },
                "model_class": {"vision": vl_cls, "text": text_cls},
                "architecture": text_architecture,
                "model_type": text_model_type,
                "error": ext.get("extract_error"),
                "translation": None,
                "usage": {},
                "translation_time_seconds": None,
                "extracted_source_text": ext.get("source_text"),
                "stages": [],
            }
            if ext.get("extract_error"):
                pass
            elif ext.get("source_text"):
                try:
                    user = TEXT_TRANSLATE_PROMPT_TMPL.format(source=ext["source_text"])
                    if backend == "vllm":
                        text, usage, elapsed = _text_generate_vllm(
                            llm=text_llm,
                            tokenizer=tokenizer,
                            user_text=user,
                            max_new_tokens=max_new_tokens_text,
                        )
                    else:
                        text, usage, elapsed = _text_generate(
                            model=text_hf,
                            tokenizer=tokenizer,
                            user_text=user,
                            max_new_tokens=max_new_tokens_text,
                        )
                    eu = ext.get("extract_usage") or {}
                    rec["translation"] = text
                    rec["usage"] = {
                        "prompt_tokens": int(eu.get("prompt_tokens") or 0)
                        + int(usage.get("prompt_tokens") or 0),
                        "completion_tokens": int(eu.get("completion_tokens") or 0)
                        + int(usage.get("completion_tokens") or 0),
                        "total_tokens": int(eu.get("total_tokens") or 0)
                        + int(usage.get("total_tokens") or 0),
                    }
                    rec["translation_time_seconds"] = round(
                        float(ext.get("extract_time_seconds") or 0) + elapsed, 6
                    )
                    rec["stages"] = [
                        {
                            "name": "vl_extract_text",
                            "model": vision_model_path,
                            "usage": ext.get("extract_usage"),
                            "time_seconds": ext.get("extract_time_seconds"),
                        },
                        {
                            "name": "text_translate_zh",
                            "model": text_model_path,
                            "usage": usage,
                            "time_seconds": round(elapsed, 6),
                        },
                    ]
                except Exception as e:
                    rec["error"] = repr(e)

            _write_per_image(
                meta_dir=meta_dir,
                translations_dir=translations_dir,
                stem=stem,
                rec=rec,
            )
            summary.append(_summary_item(ext["image_path"], per_path, translations_dir, stem, rec))
            print(
                f"[translate {i + 1}/{len(extractions)}] {stem} "
                f"tokens={rec['usage'].get('total_tokens')} "
                f"time={rec['translation_time_seconds']}s "
                f"err={'yes' if rec['error'] else 'no'}"
            )
    finally:
        if backend == "vllm":
            _cleanup(text_llm, tokenizer)
            if not unload_vision_between_stages and vl_llm is not None:
                _cleanup(vl_llm, vl_processor)
        else:
            _cleanup(text_hf, tokenizer)
            if not unload_vision_between_stages:
                try:
                    _cleanup(vl_hf, vl_processor)
                except Exception:
                    pass

    ts_base: dict[str, Any] = {
        "pipeline": "two_stage_vl_extract_then_text_translate",
        "backend": backend,
        "model": model_name,
        "vision_model_path": vision_model_path,
        "text_model_path": text_model_path,
        "text_model_class": text_cls,
        "vl_model_class": vl_cls,
        "text_architecture": text_architecture,
        "text_model_type": text_model_type,
        "count": len(image_paths),
        "max_new_tokens_vl": max_new_tokens_vl,
        "max_new_tokens_text": max_new_tokens_text,
        "dtype": dtype,
        "extract_prompt": EXTRACT_OCR_PROMPT,
        "translate_prompt_template": TEXT_TRANSLATE_PROMPT_TMPL,
        "items": summary,
    }
    if backend == "vllm":
        ts_base["vllm"] = {
            "tensor_parallel_size": vllm_tensor_parallel_size,
            "gpu_memory_utilization": vllm_gpu_memory_utilization,
            "max_model_len": vllm_max_model_len,
            "enforce_eager": vllm_enforce_eager,
            "warmup_image": vllm_warmup_image,
            "warmup_max_new_tokens": vllm_warmup_max_new_tokens,
        }
    else:
        ts_base["device_map"] = device_map
        ts_base["attn_implementation"] = attn_implementation

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(ts_base, f, ensure_ascii=False, indent=2)


def _summary_item(
    img_path: str,
    per_path: Path,
    translations_dir: Path,
    stem: str,
    rec: dict[str, Any],
) -> dict[str, Any]:
    has_t = bool(rec.get("translation"))
    u = rec.get("usage") or {}
    return {
        "image_path": img_path,
        "per_image_json": str(per_path),
        "translation_txt": str(translations_dir / f"{stem}.txt") if has_t else None,
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "total_tokens": u.get("total_tokens"),
        "translation_time_seconds": rec.get("translation_time_seconds"),
        "error": rec.get("error"),
    }


def _cleanup(*objs) -> None:
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="本地开源模型：批量图片翻译（自动识别 VL 端到端 / 两阶段文本 LM）"
    )
    parser.add_argument(
        "--model-path",
        default="/path/to/models/Qwen3-VL-8B-Instruct",
        help="目标模型权重目录（VL 或纯文本 LM）",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="输出子目录名；默认取 --model-path 最后一级",
    )
    parser.add_argument(
        "--vision-model-path",
        default="/path/to/models/Qwen3-VL-8B-Instruct",
        help="两阶段模式下用于从图片摘录原文的 VL 权重目录",
    )
    parser.add_argument(
        "--input-dir",
        default="/path/to/test1/regional",
        help="图片目录（按文件名排序）",
    )
    parser.add_argument("--limit", type=int, default=50, help="最多处理张数")
    parser.add_argument(
        "--output-root",
        default="/path/to/output/photo_translate_vlm",
        help="输出根目录",
    )
    parser.add_argument(
        "--pipeline",
        default="auto",
        choices=["auto", "vl", "two-stage"],
        help="auto（按 config 自动判别，推荐）/ vl（强制端到端）/ two-stage（强制两阶段）",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="生成上限（VL 端到端用）")
    parser.add_argument("--max-new-tokens-vl", type=int, default=2048, help="两阶段第一步")
    parser.add_argument("--max-new-tokens-text", type=int, default=2048, help="两阶段第二步")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attn-implementation",
        default=os.environ.get("HF_ATTN_IMPLEMENTATION", ""),
        help="flash_attention_2 / sdpa / eager；留空由 transformers 自选",
    )
    parser.add_argument(
        "--unload-vision-between-stages",
        action="store_true",
        help="两阶段时在载入文本模型前卸载视觉模型，省显存",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="per_image_json/<stem>.json 已完整时跳过（含两阶段的摘录复用）",
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("VLM_BACKEND", "transformers"),
        choices=["transformers", "vllm"],
        help="推理后端：transformers（HF model.generate）或 vLLM（需 CUDA + pip install vllm）",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
        help="vLLM gpu_memory_utilization（默认 0.9）",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1")),
        help="vLLM tensor_parallel_size（多卡切分）",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help="可选：覆盖 vLLM max_model_len；也可设环境变量 VLLM_MAX_MODEL_LEN",
    )
    parser.add_argument(
        "--vllm-enforce-eager",
        action="store_true",
        help="vLLM enforce_eager=True（调试用，略慢）",
    )
    parser.add_argument(
        "--vllm-warmup-image",
        default=None,
        help="仅 vLLM：模型加载后对指定图片先推理一次以预热（不写 per_image_json）；再处理 --input-dir",
    )
    parser.add_argument(
        "--vllm-warmup-max-new-tokens",
        type=int,
        default=512,
        help="预热单次 max_tokens（默认 512，减小可略加快预热）",
    )
    args = parser.parse_args()

    if args.vllm_max_model_len is None:
        vm = os.environ.get("VLLM_MAX_MODEL_LEN")
        if vm and str(vm).strip():
            try:
                args.vllm_max_model_len = int(str(vm).strip())
            except ValueError:
                pass

    if args.backend == "vllm":
        _ensure_hggc_runtime_env_for_vllm()
        _warn_if_vllm_without_cuda()

    in_dir = Path(args.input_dir).resolve()
    if not in_dir.is_dir():
        raise SystemExit(f"输入目录不存在: {in_dir}")
    exts = (".jpg", ".jpeg", ".png", ".webp", ".gif")
    paths = list_image_paths(in_dir, args.limit, exts)
    if not paths:
        raise SystemExit(f"目录中未找到图片: {in_dir} 扩展名 {exts}")

    model_name = args.model_name or Path(args.model_path.rstrip("/")).name
    out_dir = Path(args.output_root).resolve() / _safe_dirname(model_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    attn = (args.attn_implementation or "").strip() or None

    kind_auto, arch, model_type = detect_model_kind(args.model_path)
    if args.pipeline == "auto":
        kind = kind_auto
    elif args.pipeline == "vl":
        kind = "vl"
    else:
        kind = "text"

    print(
        f"检测: {args.model_path}\n"
        f"  architecture={arch} model_type={model_type} 推断={kind_auto}；实际采用 pipeline={'vl_end2end' if kind == 'vl' else 'two_stage'}"
    )

    if kind == "vl":
        run_end2end_vl(
            model_path=args.model_path,
            model_name=model_name,
            image_paths=paths,
            out_dir=out_dir,
            max_new_tokens=args.max_new_tokens,
            dtype=args.dtype,
            device_map=args.device_map,
            attn_implementation=attn,
            resume=args.resume,
            architecture=arch,
            model_type=model_type,
            backend=args.backend,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
            vllm_max_model_len=args.vllm_max_model_len,
            vllm_enforce_eager=args.vllm_enforce_eager,
            vllm_warmup_image=args.vllm_warmup_image,
            vllm_warmup_max_new_tokens=args.vllm_warmup_max_new_tokens,
        )
    else:
        run_two_stage(
            vision_model_path=args.vision_model_path,
            text_model_path=args.model_path,
            model_name=model_name,
            image_paths=paths,
            out_dir=out_dir,
            max_new_tokens_vl=args.max_new_tokens_vl,
            max_new_tokens_text=args.max_new_tokens_text,
            dtype=args.dtype,
            device_map=args.device_map,
            attn_implementation=attn,
            unload_vision_between_stages=args.unload_vision_between_stages,
            resume=args.resume,
            text_architecture=arch,
            text_model_type=model_type,
            backend=args.backend,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
            vllm_max_model_len=args.vllm_max_model_len,
            vllm_enforce_eager=args.vllm_enforce_eager,
            vllm_warmup_image=args.vllm_warmup_image,
            vllm_warmup_max_new_tokens=args.vllm_warmup_max_new_tokens,
        )

    print(f"完成。输出目录: {out_dir}")


if __name__ == "__main__":
    # vLLM EngineCore is launched in a subprocess; on Linux the default is fork.
    # If CUDA is initialized in the main process first (even implicitly), the forked
    # subprocess raises an error. Using spawn avoids inheriting the parent's CUDA
    # runtime state.
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
