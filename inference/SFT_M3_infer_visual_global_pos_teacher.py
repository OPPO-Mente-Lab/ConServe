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
"""Positive-teacher A prompt: low-resolution full image + full-score reference answer."""
from __future__ import annotations

from typing import Any

SYSTEM_PROMPT_POS_TEACHER = """图片多区域文字翻译任务。输入包含两张图，且用户消息中另有【参考答案｜正向示范】：
第一张是带橙色粗框与红色 #PID 标签的清晰完整截图，用于识别目标框内文字；
第二张是同一截图的低分辨率全局视图，用于把握页面整体主题、场景、布局和主要视觉对象；
【参考答案｜正向示范】来自高质量人工/代理评审（满分）的权威译文，用于对齐术语、语域与是否翻译。
你必须先结合第二张低分辨率全局图理解全局语境，再对照【参考答案】与第一张清晰图逐 PID 输出；禁止脱离语境的直译。
【参考答案】与第二张图仅供理解与对齐，不得把参考块说明文字原样粘贴为最终答案；须基于清晰图重新识别并按相同 JSON schema 输出。
图中有 {m} 个橙色粗框；每个框外紧邻一个红色 #PID 标签，框与标签共同标识一个独立处理单元。
按用户给出的 #PID 顺序逐框识别文字并判断是否翻译。清晰可读、非{target_lang}的自然语言必须翻译；已是{target_lang}、纯数字符号、URL或邮箱、代码标识符、单字符、商标Logo、人名或模糊残缺文字不翻译。
只输出严格单行JSON数组，长度必须为 {m}，数组元素与用户给出的 #PID 顺序一一对应；不得遗漏、合并、拆分或交换。
每个数组元素固定且只能包含以下三个字段，字段顺序固定:
需要翻译:{{"should_translate":true,"original_text":"框内原文","translated_text":"结合语境的{target_lang}译文"}}
不需翻译:{{"should_translate":false,"original_text":"框内可辨文字","translated_text":""}}
original_text只填写对应橙框内的可见文字；translated_text仅在should_translate=true时填写；false时必须为空字符串。"""


def build_user_prompt_pos_teacher(target_lang: str, pids: list[int]) -> str:
    m = len(pids)
    labels = "、".join(f"#{pid}" for pid in pids)
    order = " → ".join(f"#{pid}" for pid in pids)
    return (
        f"目标语言：{target_lang}。输入为两张图 + 正向参考答案："
        f"第一张=清晰待译图（{m} 个橙框，编号 {labels}）；"
        f"第二张=低分辨率全局语境图；"
        f"后文【参考答案｜正向示范】=满分标准译文。\n"
        f"请先结合第二张图弄清全局语境，再对照参考答案翻译第一张图；"
        f"术语与是否翻译应与参考对齐，但仍须依据清晰图重新识别原文。\n"
        f"严格按 {order} 输出长度为 {m} 的三字段JSON数组；"
        f"禁止脱离语境的直译；不要输出对第二张图或参考块的描述。"
    )


def format_pos_line(pid: int, para: dict[str, Any]) -> str:
    st = bool(para.get("should_translate"))
    orig = " ".join(str(para.get("original_text") or "").split())
    trans = " ".join(str(para.get("translated_text") or "").split())
    if st:
        return f"#{pid} | should_translate=true | original={orig} | translated={trans}"
    return f"#{pid} | should_translate=false | original={orig} | translated="


def append_pos_reference_block(
    base_user: str,
    reference_by_pid: dict[int, dict[str, Any]],
    target_pids: list[int],
) -> str:
    lines = []
    for pid in target_pids:
        para = reference_by_pid.get(pid)
        if para is None:
            continue
        lines.append(format_pos_line(pid, para))
    if not lines:
        return base_user
    order = " → ".join(f"#{pid}" for pid in target_pids)
    body = "\n".join(lines)
    return (
        f"{base_user.rstrip()}\n\n"
        f"【参考答案｜正向示范｜必须对齐，禁止原样粘贴说明】\n"
        f"{body}\n"
        f"要求：结合参考答案与低清全局图；术语/是否翻译与参考对齐；"
        f"仍须基于清晰橙框重新识别；输出顺序仍为 {order}。"
    )
