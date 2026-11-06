---
name: background-detail
description: Use when background evaluation has Major or Critical candidates, at least three `Minor` candidates, low confidence, high-risk content, or repair regions near a primary subject, scene element, icon, complex texture, gradient, or repeated pattern.
---

# Background Detail Skill

## Overview

Perform focused second-pass review of suspected background and scene damage introduced by source-text erasure or translated-text refill. Confirm or reject candidates, consolidate duplicate symptoms under one root cause, calibrate severity, and provide an actionable repair suggestion for every confirmed error.

## Natural-Language Output Policy

Follow `output_language` for every natural-language output field. When it is `zh-CN`, write `evidence`, `suggestion`, rejection explanations, and other free-text diagnoses in Simplified Chinese. Keep schema keys and enums in English. Quoted image text may remain in its original language.

## Scope

Handle smearing, local color breaks, gradient breaks, texture discontinuity, hard inpainting edges, residual halos, scene-element damage, missing non-text content, unexpected generated content, and repeated-pattern artifacts.

Do not evaluate translation semantics, fluency, glyph rendering, typography, line breaking, reading order, alignment, overflow, or wrong-region placement.

## Trigger Conditions

Use this Skill when any condition is present:

1. Any candidate is `Major` or `Critical`.
2. There are at least three `Minor` candidates in the image.
3. Any candidate confidence is below the configured threshold.
4. A candidate affects a high-risk region.
5. A repair region is close to a primary subject, meaningful scene element, icon or decoration edge.
6. A repair region contains complex texture, gradient background, or patterned background.
7. A candidate explicitly requests detailed review.

## Review Procedure

### Source-Target Delta Gate

Before confirming or updating any candidate, record `source_baseline`, `target_change`, and `translation_introduced` in `comparison`. Confirm the candidate only when the exact same localized source-target comparison shows that source-text erasure or translated-text refill newly introduced the non-text defect or clearly worsened it.

Reject source-existing perspective, crop, partial objects, panels, color bands, moire, glare, shadows, textures, and irregular geometry when the target preserves them without clear deterioration. If the target-only change cannot be stated concretely, reject the candidate rather than infer damage from the translated image alone.

### Same-Coordinate Source Counterpart Gate

For every alleged rectangle, color band, panel, shadow, blank patch, outline, or decoration, compare the exact same coordinates in the source and target. Return `comparison.source_counterpart_present` and `comparison.target_only_residue_visible`.

Reject the candidate as `source_preserved_behavior` when an equivalent source shape, boundary, fill, or patch is already present and the target does not clearly worsen it. A shorter translated text string may expose more of an existing panel or blank area; that empty space is not `inpainting_residue`.

Confirm `inpainting_residue` only when a concrete non-readable halo, ghost, outline, smear, or fragment is newly visible in the target, `target_only_residue_visible` is true, and `source_counterpart_present` is false. Text-length difference, unused horizontal space, or an assumed erasure footprint is not evidence. If the alleged residue cannot be described independently of the translated text length, reject it as `insufficient_evidence`.

### Imperceptible Surface Trace Gate

For surface repair types, reject the candidate as `insufficient_evidence` when all conditions below are true:

- `visible_at_normal_scale=false`;
- `requires_side_by_side_comparison=true`;
- the affected scope is small and `contrast_with_surroundings=low`;
- readability, scene-element, and operation impact are all `none`;
- `source_background_overwritten=false`;
- no independent non-text damage or impact tag is supported.

Such a zoom-only haze, tiny dot, weak tonal change, or suspected erasure mark is not a Minor error. A Minor must be noticeable at normal fit-to-screen scale or readily apparent in one direct side-by-side comparison.

### Repeated Rectangular Refill Policy

Inspect all translated text regions for newly introduced solid or near-solid rectangular masks. One small low-contrast block is Minor or no error. When at least two normal-scale-visible blocks share one fill cause, have medium or high contrast, overwrite continuous source background, and cumulatively affect a medium or large portion of a meaningful panel, retain one `hard_inpainting_edge / Major` or `color_discontinuity / Major`. Merge all instances under the same root cause instead of scoring each block.

For `unexpected_generated_content`, require a named target element, direct proof that the source counterpart is absent, normal-scale visibility, and bbox-evidence agreement. Reject source-existing panels, tabs, half-circles, buttons, shadows, decorations, or controls.

1. Inspect targeted candidates and every supplied `detail_risk_region`.
2. Compare the exact source and target regions at normal viewing scale and close scale.
3. Verify that the difference is newly introduced in the target, rather than source noise, compression, moire, antialiasing, or exposure variation.
4. Compare color, illumination, gradient direction, texture structure, edge geometry, and non-text scene content.
5. Identify one primary root cause per affected region and move consequences into `impact_tags`.
6. Confirm, update, or reject every targeted candidate.
7. Add a new error only when it is visually explicit, independent, and not a consequence of an existing root cause.
8. Calibrate severity from visible scene impact, recoverability, and effect on recognition, reading, operation, or trust.
9. Complete `visible_at_normal_scale`, `requires_side_by_side_comparison`, `readability_impact`, `scene_element_impact`, `operation_impact`, `affected_scope`, `rectangular_fill_visible`, `repeated_rectangular_regions`, `contrast_with_surroundings`, and `source_background_overwritten` for every updated or new error.
10. Correct bbox and confidence when evidence permits. Require `bbox_evidence_match` for every local error; reject or relocalize a candidate whose bbox points to an unrelated object or region.
11. Return strict JSON only.

## Source-Text Footprint Review

For every updated or new candidate, determine whether the defect is inside, extends beyond, or is separate from the original source-text footprint. Return the result in `comparison` using `relation_to_source_text_footprint`, `recognizable_source_glyphs`, `independent_non_text_damage`, and `independent_repair_required`.

When recognizable source glyphs remain with an underline, glow, shadow, outline, or color streak from the same source text, reject the Background candidate as `text_rendering_issue`. These are one `source_text_residue` root cause. Retain Background only for independently visible non-text damage outside the text footprint that requires a separate repair.

For a source-existing solid container such as a pill, tag, badge, banner, button background, or message bubble, treat ordinary content-driven resizing as valid when it fits translated text and preserves the container's visual role. Reject a smearing candidate when width or height is the only change and the rounded boundary, fill containment, and surrounding scene remain intact. A source pattern being hidden beneath the resized container is not damage by itself.

Confirm a background defect only when the rounded boundary is lost or malformed, fill color leaks outside the container, the repair overwrites unrelated content, or pixels extend beyond the functional container without a text-fit reason. Route geometry-only problems to Layout.

## Moire and Screen-Capture Artifact Policy

Start with falsification: check whether the candidate is only a difference in moire, scan lines, glare, compression, refresh bands, or another photographed-display artifact. Return `capture_artifact_kind`, `capture_artifact_preservation`, `independent_non_text_damage`, and `independent_scene_content` in `comparison`.

Missing or weakened capture noise is Minor by default. Reject it when the source artifact is actually preserved. It can remain Major only when direct source-target evidence shows independent material damage to meaningful non-text scene content; name that content in `independent_scene_content`.

Use only these impact tags: `local_scene_degradation`, `large_area_corruption`, `content_missing`, `false_content_introduced`, `primary_subject_unrecoverable`, `key_icon_unrecognizable`, `critical_region_untrustworthy`, and `scene_unusable`. Do not invent another tag.

When several spatially separate patches in one panel come from the same fill operation and require the same repair, retain one primary error and list the other candidate IDs in `root_cause_groups.merged_error_ids`. Do not preserve both a panel-wide error and its local duplicate patches.

## Severity Refinement

For texture candidates, complete `comparison.texture_evidence`: `source_primitive`, `source_primitive_visible`, `target_break_visible`, and `visible_at_normal_scale`. The source primitive must name and localize a concrete pattern, edge, strand, line, fiber, or motif. Generic claims such as "the texture looks smoother", "detail may be missing", or "the area differs" are insufficient for Major; cap them at Minor.

For scene-element damage or missing-content candidates, complete `comparison.scene_element_evidence`: `source_element`, `source_element_visible`, `target_element_missing_or_malformed`, and `functional_loss`. If the element remains visible and functional, reject the candidate rather than lowering it to a cosmetic error.

Detectability is not severity. Use `Minor` for a small or cosmetic defect that remains noticeable at normal fit-to-screen scale or readily apparent in one direct side-by-side comparison, without material reading, operation, scene-element, or large-area impact. A zoom-only trace that meets the Imperceptible Surface Trace Gate is no error.

For surface repair types, retain `Major` only when the defect is clearly visible at normal viewing scale without side-by-side comparison and also has material readability, operation, meaningful-element, or large-area impact. If either condition is missing or uncertain, downgrade to `Minor`.

A high-contrast rectangular refill is a narrow `Major` exception when it is clearly visible at normal scale, forms a new solid or near-solid rectangle, has strong light-dark contrast against the immediately adjacent background, and source-target comparison confirms that it overwrites a previously continuous source background or non-text scene area. This applies even when text remains readable. Do not apply the exception to mild color shifts, exposure differences, glare, compression, source-existing rectangles, or zoom-only evidence.

Do not apply the rectangular-refill exception to source glyphs, underlines, text shadows, glows, or color streaks confined to the original text footprint. A Major claim must identify independently overwritten non-text pixels outside that footprint.

Use `Critical` only for unusable large-area corruption, substantial primary-subject damage, unrecognizable key icons, false generated content that changes scene interpretation, or damaged safety, payment, or price regions that become unreadable or untrustworthy.

Neither a large bbox, a complex texture, a gradient, a subject edge, nor a high-risk label is automatically `Critical`.

Do not upgrade an existing solid container because content-driven resizing makes it larger or hides more of the same background pattern. `Major` requires clear new damage outside the container, substantial boundary destruction, or material damage to unrelated scene content.

## Suggestion Policy

Every confirmed updated or new error must provide an actionable repair suggestion in `output_language`. Name the affected region and specify one direct repair, such as reconstructing the source texture direction, matching surrounding color, continuing a gradient, removing a halo, restoring an object edge, or deleting unsupported generated content.

Do not rewrite translated wording or prescribe layout changes.

## References

Read the files in `references/` for taxonomy, severity boundaries, attribution rules, root-cause consolidation, examples, and the strict output schema.

## Output Contract

Return `updated_errors`, `new_errors`, `rejected_errors`, `root_cause_groups`, and `notes` exactly as defined in `references/output-schema.md`. Do not return scores, penalties, another Skill plan, or prose outside JSON.
