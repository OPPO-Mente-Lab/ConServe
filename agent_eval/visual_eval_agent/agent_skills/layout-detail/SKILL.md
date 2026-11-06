---
name: layout-detail
description: Use when translated-image layout has Major or Critical candidates, at least three Minor candidates, low confidence, high-risk content, uncertain region correspondence, or suspected position, alignment, overflow, line-break, reading-direction, reading-order, crowding, occlusion, or wrong-region defects.
---

# Layout Detail Skill

## Overview

Refine only `visual.layout` candidate errors by inspecting the same original and translated image pair. Confirm, reject, reclassify, relocate, severity-adjust, and group candidates by root cause. Provide an actionable repair suggestion for every confirmed updated or new error.

## Natural-Language Output Policy

Follow `output_language` for every natural-language output field. When it is `zh-CN`, write `evidence`, `suggestion`, rejection explanations, and other free-text diagnoses in Simplified Chinese. Keep schema keys and enums in English. Quoted image text may remain in its original language.

## Scope

Handle text-region correspondence, position, alignment, container fit, line breaking, reading direction, reading order, spacing, crowding, non-text occlusion, and wrong-region placement.

Do not judge translation meaning, target-language fluency, glyph-pixel quality, typography style, source-text residue, or background repair. Route those issues to the appropriate Agent.

## Trigger Conditions

Use this Skill when any condition is present:

1. A candidate is `Major` or `Critical`.
2. The image has at least three `Minor` Layout candidates.
3. Candidate confidence is below the configured threshold.
4. A candidate affects a high-risk region.
5. Source-to-target region correspondence is uncertain.
6. Natural reading direction or reading order is uncertain.
7. A candidate explicitly requests detailed review.
8. The combined visual orchestrator supplies
   `review_focus=internal_content_canvas_overflow` after Text Rendering detects a
   conspicuous source-target font-size inconsistency. In this focused mode, run
   the Internal Content-Canvas Boundary Coverage Pass even when the Layout Base
   Agent returned no candidates. Add a new `text_overflow` error only when the
   target visibly crosses an internal screen, page, slide, dialog, or panel
   boundary relative to the source.

## Root Cause Procedure

### Source-Target Delta Gate

Before confirming or updating any candidate, record `source_baseline`, `target_change`, and `translation_introduced` in `comparison`. Confirm the candidate only when the same localized source-target comparison shows that translation, erasure, or refill newly introduced the defect or clearly worsened it.

Reject source-existing alignment, whitespace, rotation, edge proximity, perspective crop, partial framing, and other unusual composition when the target preserves the same state without clear deterioration. Image-edge proximity is not clipping. A crop or overflow claim requires actual visible glyph loss newly present in the target; a bbox touching the edge or a photographed display already leaving the source frame is insufficient.

### Container Alignment Coverage Pass

For title bars, speech bubbles, message cards, buttons, labels, input fields, tabs, badges, menus, and table cells, compare source and target centering, edge padding, alignment mode, and whether the first and last glyph remain inside the intended container. Repeat the check across repeated same-role elements such as message cards and `Read more...` links. Preserve link-token continuity, punctuation adjacency, and baseline.

When several repeated instances share the same root cause, retain one error and list all affected regions in evidence or the root-cause group. When alignment and line breaking come from one text-block format setting, retain one dominant Layout error rather than scoring both symptoms.

### Adjacent-Region Collision Coverage Pass

Inspect all four sides of every targeted translated block for newly introduced contact with neighboring titles, labels, preserved text, buttons, icons, window chrome, or text rows. Complete `comparison.collision_evidence` with `translated_block`, `overlapped_neighbor`, `source_separation_preserved`, `visible_intersection`, `readability_impact`, and `operation_impact`.

Confirm Layout `text_overflow` when the source had visible separation but target glyphs newly cross their intended container and visibly intersect an adjacent region at normal scale. A small localized collision with both items still readily readable is `Minor`; use `Major` when a prominent title, field, neighboring text, or control becomes materially hard to read or use. Do not omit a visible collision merely because its intended words remain guessable.

If the same pixels contain two text layers after placement is conceptually corrected, route the independent glyph-layer defect to Text Rendering. When restoring the block's container fit removes the overlap, retain Layout as the root cause and merge Text Rendering overlap as a secondary impact.

### Internal Content-Canvas Boundary Coverage Pass

Treat a visible screen, page, slide, document sheet, dialog, panel, card, or other framed content surface as an internal content canvas. Compare prominent source and target text near all four internal boundaries, independently of the outer photograph boundary.

This check is mandatory for photographed displays, device screens, pages, slides, dialogs, and panels. Trace high-contrast bezels, page edges, slide edges, dialog borders, and panel separators, then inspect every prominent title or body block within one text-height of those boundaries.

Complete `comparison.content_canvas_boundary` with `canvas_type`, `source_inside_canvas`, `target_crosses_canvas`, `target_clipped_or_outside`, `reading_flow_impact`, and `content_hidden`.

Confirm `text_overflow / Major` when a prominent title or meaningful body block was inside the source content canvas but newly crosses into a bezel, outer photograph, neighboring panel, clipped region, or unrelated surface in the target, and the reading flow or usable content is materially damaged. A tiny intact padding deviation remains Minor.

If the same oversized or displaced block also produced a `text_region_crowding` candidate, merge that symptom into the `text_overflow` root cause and retain one scored error.

Text visibly painted across a dark bezel, outside a page edge, or into a neighboring panel proves `target_crosses_canvas=true` even when OCR returns one intact bbox and the string remains readable.

1. Inspect targeted candidate regions before expanding to surrounding context.
2. Compare translated text, intended container, source region, and nearby visual elements.
3. For direction candidates, compare the source and target page rotation, block axes, line progression, and block order in image coordinates. Return `source_page_rotation`, `target_page_rotation`, `source_block_axis`, `target_block_axis`, `source_arrangement_preserved`, `new_rotation_delta_degrees`, and `new_head_rotation_required` in `comparison`. Also return `orientation_anchor_evidence` using the same visible Latin text, number, punctuation, container axis, or UI geometry in both images. Set `cjk_glyph_shape_only` to true when the claim relies only on square CJK glyph appearance; that evidence cannot establish orientation. Equivalent source and target rotation is correct even when both require head or device rotation.
4. Decide whether the candidate is a Layout defect or belongs to Text Rendering or Background.
5. Identify one causal defect for each affected region.
6. Move secondary visible consequences into `impact_tags`.
7. Confirm, reject, or update every targeted candidate.
8. Add a new error only when it is visually explicit, independent, and not a consequence of an existing root cause.
9. Calibrate severity from observable reading and usability impact.
10. Correct `bbox_1000` and confidence when evidence permits, then return strict JSON only.

Before confirming a local error, require bbox-evidence agreement: the bbox must cover the exact text or container described by `evidence`. Reject or relocalize a candidate whose bbox points to an unrelated object, background region, hand, or neighboring control.

## Severity Refinement

A one-word split cannot be Major by itself. For every line-break candidate, complete `comparison.line_break_evidence`: `scope`, `characters_complete`, `reading_order_preserved`, `content_association_preserved`, `overlap_or_clipping`, and `material_reconstruction_required`. Cap the error at Minor when the scope is one word or a short phrase and all content remains complete, ordered, associated, unclipped, and recoverable without reconstruction.

### Target-Only Recoverability Definition

Judge recoverability from the target image alone. A mapping, association, reading order, or content unit is recoverable only when an ordinary viewer can determine it from visible target geometry and target text without borrowing source wording, expected translation, nearby semantic context, or external knowledge.

### Control-Mapping Critical Gate

Confirm the most specific Layout root cause as Critical when two or more control mappings are lost, or when one complete control group loses its mapping, and the target image alone does not show which answer, label, field, button, blank, or option belongs to which control. One shifted label whose control remains obvious is not Critical.

### Structured-Association Critical Gate

Confirm Critical when translated content crosses or collapses at least two structural boundaries and at least two structured associations become unrecoverable. Count row-column, field-value, label-control, date-result, identifier-value, and equivalent table or form relationships. Do not require every row or the whole page to fail.

### Primary-Content Crowding Critical Gate

Confirm `text_region_crowding / Critical` when at least two separated source regions collapse into one target area, multiple logical rows or groups overlap or interleave, and at least two content units require guessing to determine reading order or ownership from the target image alone. Localized readable overlap remains Minor or Major.

Use `Minor` when reading remains effortless relative to the source and the issue is slight. Use `Major` when the target newly makes users reread, search for order, inspect carefully, turn their head, or when a meaningful region is visibly disrupted but recoverable. Use `Critical` only for wrong content association, unavailable key information, unrecoverable primary content, or an unusable main page task.

Use `text_region_crowding / Critical` when multiple source regions collapse into one target area and the main translated content becomes unreadable. Treat Text Rendering overlap as a secondary consequence when correcting placement would resolve it.

Explicitly preserve Critical candidates when answer choices, field labels, or button text lose their control mapping; when core table rows or columns collapse and their associations cannot be recovered; or when large-scale target-only crowding makes the primary reading order unrecoverable at normal scale. State the corresponding source arrangement, target-only change, normal-scale visibility, and concrete unusable or untrustworthy consequence. The Visual Critic applies the final Critical Gate.

Reject `reading_direction_mismatch` when source and target preserve equivalent reading axes and glyph orientations. A rotated source that remains equally rotated after translation is not a layout defect.

For a direction Major, `orientation_anchor_evidence` must show a shared non-CJK anchor visible in both images and set `anchor_mismatch_confirmed` to true. Without that evidence, cap the candidate at Minor and request human review. When the shared anchor has equivalent source and target rotation, reject the candidate.

No error type, large box, high-risk label, overflow, or occlusion is automatically `Critical`.

## Suggestion Policy

Every confirmed updated or new error must include a non-empty `suggestion` in `output_language`. Identify the affected region and provide one directly usable layout repair, such as moving the block, restoring natural orientation, changing line breaks, resizing the container, or clearing protected content.

Do not rewrite translation semantics or prescribe background reconstruction.

## Output Contract

Return `updated_errors`, `new_errors`, `rejected_errors`, `root_cause_groups`, and `notes` as defined in `references/output-schema.md`. Do not return scores, general commentary, or another Skill plan.

## References

Read the references needed for the current case:

- Fixed types: `references/taxonomy.md`
- Severity decisions: `references/severity-rubric.md`
- One-root-cause rules: `references/root-cause-rules.md`
- Cross-Agent attribution: `references/boundary-rules.md`
- General examples: `references/examples.md`
- Strict response fields: `references/output-schema.md`

## Self-check

Verify that every targeted candidate is updated, rejected, or intentionally preserved; one causal defect is not scored multiple times; each confirmed error has direct visual evidence, valid normalized localization, allowed impact tags, calibrated severity, confidence, and a concrete suggestion; and the response contains JSON only.
