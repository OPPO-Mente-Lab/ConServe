---
name: text-coverage-detail
description: Refines uncertain source-to-target text coverage when many clear source blocks are unmatched but the deterministic large-omission gate cannot be safely applied.
---

# Text Coverage Detail Skill

## 目的

仅检查原图中可翻译的可见文字是否在译图中获得了对应覆盖。该 Skill
处理覆盖率边界情况，不评估译文措辞、语法、流利度、字体、排版或背景。

## 触发条件

满足以下任一情况时可以调用：

- 清晰源文字存在较多未匹配块，但未达到确定性 Critical 门槛；
- OCR 置信度、文字可见性或文字是否需要翻译存在疑问；
- 多个源文字块可能被合并为一个目标文字块；
- 目标图存在模糊文字，无法判断是漏译还是识别失败；
- 主要文字区域是否完整覆盖无法由结构化结果确定。

证据已经同时满足确定性 Critical 门槛时不得调用本 Skill，直接保留
`omission / Critical`。

## 工作步骤

1. 分别检查原图和译图，不得用另一张图片补全不可见文字。
2. 核对候选未匹配源块是否属于可翻译、对用户可见的内容。
3. 排除网址、纯标识符、品牌名、型号和无需翻译的装饰字符。
4. 检查未匹配源块是否实际被目标图中的合并文本覆盖。
5. 按连续区域和共同根因合并，不得按每个 OCR 框重复扣分。
6. 给出确认错误、非错误或仍不确定的结论。

## 严重度

- `Minor`：一个低重要度短标签缺失，不影响主要阅读任务。
- `Major`：一个有意义的标题、字段、句子或局部正文缺失，但主要任务仍可完成。
- `Critical`：主要正文整体消失，或绝大多数可翻译内容没有对应译文，导致图片主要信息无法使用。

不得仅根据 bbox 数量、bbox 面积或源目标字符数差异判定 Critical。

## 输出

只返回 `references/output-schema.md` 定义的 JSON。所有诊断说明和修改建议使用简体中文。

