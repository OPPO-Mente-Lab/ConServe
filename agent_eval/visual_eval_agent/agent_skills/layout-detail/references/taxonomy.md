# Layout Taxonomy

Use exactly one primary type per independent causal defect.

| Type | Use when |
| --- | --- |
| `position_shift` | The translated block is displaced from its intended location. |
| `alignment_error` | The block is in the correct container but its alignment is visibly wrong. |
| `text_overflow` | Text extends outside its intended container without a more direct causal displacement. |
| `unreasonable_line_break` | Line division damages phrase grouping, syntax, rhythm, or creates conspicuous fragments. |
| `reading_direction_mismatch` | Horizontal, vertical, or rotated orientation conflicts with natural viewing and reading. |
| `reading_order_error` | The sequence of lines or separate blocks cannot be followed naturally. |
| `text_region_crowding` | Spacing between characters, lines, or blocks is excessively crowded. |
| `content_occlusion` | Text directly covers non-text visual content without another observable Layout defect causing it. |
| `wrong_region_fill` | Translation is placed in a container belonging to different source content. |

## Orientation diagnostic

In true Chinese vertical typesetting, individual glyphs remain upright and their positions advance from top to bottom. A horizontal phrase whose glyphs or baseline are rotated sideways is `reading_direction_mismatch`, even when it also appears poorly centered.

CJK glyph shape alone cannot establish orientation. Confirm direction using the same visible Latin text, number, punctuation, container axis, or stable UI geometry in source and target, and record it in `orientation_anchor_evidence` with `anchor_mismatch_confirmed`.

Do not reduce a sideways-orientation defect to alignment or position. Use alignment only when glyph orientation and reading direction are already natural.

Allowed `impact_tags`:

- `text_overflow`
- `content_occlusion`
- `requires_head_rotation`
- `disrupted_reading_flow`
- `rereading_required`
- `partial_content_hidden`
- `wrong_content_association`
- `key_content_hidden`
- `primary_content_unrecoverable`
- `page_unusable`

Do not invent synonyms. If the base Agent used `rotated_text`, normalize it to `reading_direction_mismatch`.
