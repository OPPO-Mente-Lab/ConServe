# Text Rendering Boundary Examples

## Clear text split into many lines

Translated glyphs are sharp and complete, but the sentence is fragmented into many short lines.

Decision: Reject the Text Rendering candidate. Route to Layout as unreasonable line breaking or reading order.

## Local text overlap

Two text layers overlap in one heading. Three characters require careful inspection, but the phrase remains recoverable.

Decision: Confirm `overlapping_text / Major`. Do not upgrade solely because overlap exists.

## Faint source residue

A pale source-script stroke remains outside the translated glyphs without slowing reading.

Decision: Reject unless the source fragment is identifiable and the residue is visible at normal scale. Zoom-only or bbox-guided suspicion is not an error.

## Source-preserved inactive label

The source uses a pale grey label to show an inactive state. The translated result retains the same pale grey state and remains as readable as the source.

Decision: Reject `font_color_mismatch` as `source_preserved_behavior` or insufficient evidence. Do not make the target darker merely because it has low contrast.

## Ambiguous comparison-only residue

A possible source stroke appears only after zooming and comparing the image pair. It is not unambiguously a source character at normal viewing scale and does not interfere with the target glyphs.

Decision: Reject. Residue that is not visible at normal scale does not meet the evidence gate.

## Residue mixed with translated text

Dark source characters remain under the translated text and several target glyphs cannot be separated visually.

Decision: Usually `source_text_residue / Major`; use Critical only if primary or high-risk text becomes unreadable.

## Background smear

All target glyphs are crisp, but the repaired texture behind them has a visible smear.

Decision: Reject Text Rendering. Route to Background.

## Intact text outside a box

Every character is clear, but the final line extends outside its intended panel.

Decision: Reject Text Rendering. Route to Layout.

## Systemic garbling

Most primary translated text consists of malformed or unrecognizable target-script glyphs.

Decision: Confirm `garbled_text / Critical`.

## Missing protected values

The source contains a visible student identifier and email address. The translated image leaves both corresponding value regions blank while preserving the surrounding form.

Decision: Confirm `missing_text`. Use Critical when the missing values make the identity or contact form materially untrustworthy; otherwise use Major.
