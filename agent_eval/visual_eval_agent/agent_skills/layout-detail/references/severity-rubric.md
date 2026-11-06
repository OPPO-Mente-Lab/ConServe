# Layout Severity Rubric

Judge outcome, not type name.

## Decision order

1. Can the intended region and reading sequence be identified reliably?
2. Can users read normally without changing posture or repeatedly reconstructing order?
3. Is the affected content recoverable?
4. Does the defect change content association or prevent a key action?

## Minor

- Correct region and unchanged reading order.
- No need to turn the head, rotate the device, or reread.
- Slight position, alignment, spacing, or line-break issue.
- No meaningful content is hidden.
- A one-word or short-phrase split is Minor at most when characters, order, association, and readability remain intact.

## Major

- Requires rereading, careful inspection, or searching for the next line or block.
- Requires users to turn their head or rotate the device.
- Clearly disrupts a meaningful region through displacement, overflow, crowding, or occlusion.
- Content remains reliably recoverable and the main task remains usable.

Major requires material reconstruction, repeated broken flow, overlap or clipping, wrong association, or a newly introduced source-relative viewing burden. Source and target that preserve the same page rotation and block axis are not Major. Differences caused only by target-script glyph conventions are not errors.

For `reading_direction_mismatch`, CJK glyph shape alone cannot establish orientation. A Major requires `orientation_anchor_evidence` from a shared Latin token, number, punctuation mark, container axis, UI geometry, or mixed anchor that is visible in both images and confirms a source-target rotation change. Missing or ambiguous anchor evidence is Minor at most; equivalent anchor rotation rejects the direction claim.

## Critical

Critical requires at least one concrete consequence:

- `wrong_content_association`
- `key_content_hidden`
- `primary_content_unrecoverable`
- `page_unusable`

Use Critical only when the evidence directly supports that consequence. High risk, large area, overflow, or occlusion alone is insufficient.

Multiple source regions collapsed into one target area are `text_region_crowding / Critical` when the main translated content becomes unreadable at normal scale. The unreadable Text Rendering consequence is not a second scored error when restoring placement resolves it.

When evidence is ambiguous, choose the lower severity, reduce confidence, and preserve the uncertainty in evidence.
