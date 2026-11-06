# Copyright 2026 OPPO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Negative-teacher B prompt: show the teacher "wrong translation examples + the offending fields + reasons + suggested fixes"."""
from __future__ import annotations

from typing import Any

SYSTEM_PROMPT_NEG_TEACHER = """图片多区域文字翻译任务。输入包含两张图，且用户消息中另有【错误翻译示例】：
第一张是带橙色粗框与红色 #PID 标签的清晰完整截图，用于识别目标框内文字；
第二张是同一截图的低分辨率全局视图，用于把握页面整体主题、场景、布局和主要视觉对象；
【错误翻译示例】提供：当前错误的 translated_text、出错字段（通常是 translated_text 中的片段）、错误原因、以及可选的建议修正。
你的任务是：理解错误示例后，基于清晰图重新翻译，输出纠正后的结果；严禁照抄错误的 translated_text。
图中有 {m} 个橙色粗框；每个框外紧邻一个红色 #PID 标签，框与标签共同标识一个独立处理单元。
按用户给出的 #PID 顺序逐框识别文字并判断是否翻译。清晰可读、非{target_lang}的自然语言必须翻译；已是{target_lang}、纯数字符号、URL或邮箱、代码标识符、单字符、商标Logo、人名或模糊残缺文字不翻译。
只输出严格单行JSON数组，长度必须为 {m}，数组元素与用户给出的 #PID 顺序一一对应；不得遗漏、合并、拆分或交换。
每个数组元素固定且只能包含以下三个字段，字段顺序固定:
需要翻译:{{"should_translate":true,"original_text":"框内原文","translated_text":"纠正后的{target_lang}译文"}}
不需翻译:{{"should_translate":false,"original_text":"框内可辨文字","translated_text":""}}
original_text只填写对应橙框内的可见文字；translated_text仅在should_translate=true时填写；false时必须为空字符串。
不要把错误原因或建议修正原文写入输出字段。"""


def build_user_prompt_neg_teacher(target_lang: str, pids: list[int]) -> str:
    m = len(pids)
    labels = "、".join(f"#{pid}" for pid in pids)
    order = " → ".join(f"#{pid}" for pid in pids)
    return (
        f"目标语言：{target_lang}。输入为两张图 + 错误翻译示例："
        f"第一张=清晰待译图（{m} 个橙框，编号 {labels}）；"
        f"第二张=低分辨率全局语境图；"
        f"后文【错误翻译示例】=错误译文、出错字段、错误原因、建议修正。\n"
        f"请先结合第二张图理解语境，再根据错误示例规避问题，基于清晰图输出纠正后的三字段JSON；"
        f"禁止照抄错误译文。\n"
        f"严格按 {order} 输出长度为 {m} 的三字段JSON数组。"
    )


def _format_one_error(err: dict[str, Any]) -> str:
    """Each error contains only three fields: target_fragment / evidence / suggestion."""
    frag = " ".join(str(err.get("target_fragment") or "").split())
    ev = " ".join(str(err.get("evidence") or "").split())
    sug = err.get("suggestion")
    sug_s = " ".join(str(sug).split()) if sug else ""
    if frag:
        bad_field = f"translated_text（问题片段: 「{frag}」）"
    else:
        bad_field = "translated_text"
    lines = [
        f"  出错字段: {bad_field}",
        f"  错误原因: {ev}" if ev else "  错误原因: （未提供）",
    ]
    if sug_s:
        lines.append(f"  建议修正: {sug_s}")
    else:
        lines.append("  建议修正: （无，请根据错误原因自行改写 translated_text）")
    return "\n".join(lines)


def format_neg_block_for_pid(pid: int, para: dict[str, Any]) -> str:
    orig = " ".join(str(para.get("original_text") or "").split())
    bad = " ".join(str(para.get("translated_text") or "").split())
    errs = para.get("errors") or []
    head = (
        f"#{pid}\n"
        f"  original_text: {orig}\n"
        f"  错误的 translated_text: {bad}"
    )
    if not errs:
        return (
            f"{head}\n"
            f"  出错字段: translated_text\n"
            f"  错误原因: （未提供细粒度评语）\n"
            f"  建议修正: （无，请自行改进）"
        )
    body = "\n".join(_format_one_error(e) for e in errs)
    return f"{head}\n{body}"


def append_neg_critique_block(
    base_user: str,
    critique_by_pid: dict[int, dict[str, Any]],
    target_pids: list[int],
) -> str:
    blocks: list[str] = []
    crit_pids = []
    for pid in target_pids:
        para = critique_by_pid.get(pid)
        if para is None:
            blocks.append(f"#{pid}\n  （无错误示例，请按常规正确翻译）")
            continue
        crit_pids.append(pid)
        blocks.append(format_neg_block_for_pid(pid, para))
    if not crit_pids:
        return base_user
    order = " → ".join(f"#{pid}" for pid in target_pids)
    body = "\n\n".join(blocks)
    return (
        f"{base_user.rstrip()}\n\n"
        f"【错误翻译示例｜供纠错，禁止照抄错误的 translated_text】\n"
        f"{body}\n\n"
        f"请根据「出错字段 / 错误原因 / 建议修正」输出纠正后的 "
        f"should_translate、original_text、translated_text；顺序仍为 {order}。"
    )
