---
name: text-pairing-detail
description: Refines image text extraction, reading order, semantic grouping, and source-target alignment when the preprocessor is uncertain or the layout is structurally complex.
---

# Text Pairing Detail Skill

## Purpose

Reinspect exactly two images and replace the candidate extraction with a more reliable `source_blocks`, `target_blocks`, and `evaluation_units` structure.

## Scope

- Correct visible-text transcription only when the pixels provide direct evidence.
- Correct normalized bboxes, reading order, block type, visibility, and confidence.
- Split unrelated regions and merge only semantically continuous visual line breaks.
- Correct one-to-one, one-to-many, many-to-one, or many-to-many alignment.
- Preserve unresolved blocks in the unmatched arrays and explain uncertainty in warnings or notes.

不得评价翻译质量，不得输出 Faithfulness 或 Fluency 错误，不得给出质量分数或严重程度。

## Procedure

1. Read the trigger reasons and inspect only the relevant structural risks first.
2. Independently verify text on the source image and translated image.
3. Rebuild the complete extraction object, not a partial patch.
4. Check every block reference and every join operation.
5. Return strict JSON only.

## Output

Return the complete extraction object with these top-level fields:

- `source_blocks`
- `target_blocks`
- `evaluation_units`
- `unmatched_source_block_ids`
- `unmatched_target_block_ids`
- `reading_order_uncertain`
- `complex_layout`
- `notes`

## References

- Grouping and alignment boundaries: `references/boundary-rules.md`
- Exact JSON contract: `references/output-schema.md`
- Varied structural examples: `references/examples.md`
