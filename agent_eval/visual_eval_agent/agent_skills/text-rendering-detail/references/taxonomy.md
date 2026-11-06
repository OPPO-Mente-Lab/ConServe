# Text Rendering Taxonomy

Use the most specific type that describes the visible rendering defect.

| Type | Use when | Do not use when |
|---|---|---|
| `blurred_text` | Soft or smeared glyph edges reduce recognition. | The background alone is smeared. |
| `jagged_text` | Glyph edges are visibly aliased or stair-stepped. | The font is merely stylistically different. |
| `too_small_to_read` | Corresponding source text is readable, but translation newly reduces its size until a concrete target fragment cannot be read at the same normal scale. | Source text is already equivalently small, target text remains readable, or the claim comes from page density or review-thumbnail scale; return no error. |
| `cropped_text` | Glyph pixels are visibly clipped. | Intact text merely crosses a container boundary; that is Layout. |
| `missing_text` | A clearly corresponding source text region or protected visible value is entirely absent in the target, with no legitimate relocation or replacement. | Only some glyphs inside a present target region are absent; use `missing_glyph`. |
| `missing_glyph` | Characters inside a present text region are absent or replaced by blanks/placeholders. Consecutive font-fallback boxes that make a meaningful line unrecoverable are Critical. | Source meaning was omitted but no visual text defect is visible. |
| `broken_stroke` | A visible character has fractured or incomplete strokes. | The entire glyph is corrupted; use `garbled_text`. |
| `overlapping_text` | Text layers or glyph groups occupy the same pixels and interfere with recognition. | Lines are close or numerous without actual overlap; that is Layout. |
| `source_text_residue` | A continuous source-script fragment is directly noticeable at normal scale without bbox guidance and remains visibly mixed with or near translated text. | Intentionally retained labels, isolated letters, narrow edge slivers, or marks noticed only after zoom/bbox guidance; return no error. |
| `garbled_text` | Glyphs are corrupted or unrecognizable as normal target-script text. | Wording is linguistically nonsensical but glyphs render normally; that is Fluency. |
| `font_mismatch` | Font character is conspicuously inconsistent with the local design. | A different writing system naturally requires different glyph forms. |
| `font_size_inconsistency` | Corresponding source text or same-level source labels use a consistent size, but translated counterparts do not. | The main consequence is unreadability; use `too_small_to_read`. |
| `font_weight_mismatch` | Bold, regular, or light weight differs from the corresponding source text without a hierarchy reason. | Outline or shadow differs while weight is preserved; use `stroke_shadow_mismatch`. |
| `font_color_mismatch` | Corresponding text color changes conspicuously without a contrast or hierarchy reason. | Background color continuity is broken; that is Background. |
| `stroke_shadow_mismatch` | Outline or shadow is conspicuously inconsistent or harms recognition. | Weight alone differs; use `font_weight_mismatch`. |

## Primary Attribution

- Character pixels are defective: Text Rendering.
- Character pixels are intact but spatial arrangement is defective: Layout.
- Character pixels are intact but repaired scene pixels are defective: Background.

One repair action should normally correspond to one primary error. Do not report several types for the same pixels unless the defects are independent and require different repairs.

Treat small differences from compression, antialiasing, camera exposure, or color profiles as non-errors. When a color mismatch also creates low contrast or changes a semantic color state, keep only the most consequential primary text-rendering error.

For repeated buttons, tabs, labels, options, or badges, compare every visible member with its corresponding source member. When one shared rendering setting makes several labels lose the same weight, size, color, or effect, retain one grouped typography error with one shared cause and a bbox covering the affected labels. Do not stop after inspecting only one member, and do not create one error per label when one repair action fixes the group.
