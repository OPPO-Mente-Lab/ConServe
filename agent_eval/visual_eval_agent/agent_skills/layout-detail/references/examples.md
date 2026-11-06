# Layout Detail Examples

## Comic: rotated dialogue

Observation: The comic is viewed upright and the speech bubble supports a top-to-bottom flow. Individual glyphs should remain upright, but a horizontal Chinese phrase and its baseline are rotated ninety degrees, so readers must turn their head.

Decision: Confirm `reading_direction_mismatch / Major` with `requires_head_rotation` and `disrupted_reading_flow`. Suggest keeping Chinese glyphs upright and arranging them naturally within the vertical bubble.

Do not reduce a sideways-orientation defect to alignment merely because the rotated phrase is also off center.

## Document: slight heading offset

Observation: A translated heading is slightly left of the source alignment but remains inside its column and is effortless to scan.

Decision: `alignment_error / Minor`.

## Poster: displacement with consequences

Observation: A translated slogan is shifted downward, crosses its banner, and covers part of an illustration.

Decision: One `position_shift` error. Add `text_overflow` and `content_occlusion`; do not create three scored errors.

## Subtitle: broken phrase grouping

Observation: A short sentence is divided into several one- or two-character lines, forcing repeated reconstruction while the glyphs remain clear.

Decision: `unreasonable_line_break / Major` with `rereading_required`.

## Interface: wrong control

Observation: The confirmation label is placed inside the cancellation control, changing the apparent action.

Decision: `wrong_region_fill / Critical` with `wrong_content_association` when the association is visually clear.

## Font style only

Observation: The block is correctly positioned and ordered, but its font does not match nearby text.

Decision: Reject Layout. Route to Text Rendering.

## Damaged texture only

Observation: The translated caption is correctly placed, but the background behind it has smearing and a broken pattern.

Decision: Reject Layout. Route to Background.
