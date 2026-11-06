# Boundary Rules

## Merge only when all are true

- The blocks belong to the same visual container or semantic item.
- They share a consistent local reading direction.
- They are consecutive in reading order.
- The next block is a continuation rather than a separate label, control, cell, caption, or speaker.

## Keep separate by default

- Heading and body text.
- Distinct buttons, labels, prices, fields, or navigation items.
- Different comic speech bubbles or speakers.
- Different table cells, columns, or list items.
- Watermarks, captions, warnings, and unrelated body text.

## Alignment evidence order

Use semantic content, container role, local reading order, and spatial region together. Center distance alone is insufficient. If evidence conflicts, lower `pair_confidence` and keep blocks unmatched instead of forcing a pair.

## Transcription boundary

Copy only visible characters. Do not infer hidden characters, silently correct source spelling, or borrow text from the other image.
