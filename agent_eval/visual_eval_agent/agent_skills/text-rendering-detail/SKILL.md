---
name: text-rendering-detail
description: Use when translated-image text has Major or Critical rendering candidates, at least three Minor candidates, low confidence, high-risk content, OCR disagreement, or uncertain blur, clipping, missing glyphs, residue, garbling, overlap, typography, color, outline, or shadow defects.
---

# Text Rendering Detail Skill

## Overview

Refine only `visual.text_rendering` candidate errors by inspecting the same original and translated image pair. Confirm, reject, reclassify, relocate, or severity-adjust each candidate and provide an actionable repair suggestion for every confirmed error.

## Natural-Language Output Policy

Follow `output_language` for every natural-language output field. When it is `zh-CN`, write `evidence`, `suggestion`, rejection explanations, and other free-text diagnoses in Simplified Chinese. Keep schema keys and enums in English. Quoted image text may remain in its original language.

## Scope

Handle character-level readability, whole-region `missing_text`, glyph completeness, source-text residue, garbled rendering, text-on-text interference, and conspicuous font, `font_size_inconsistency`, `font_weight_mismatch`, `font_color_mismatch`, outline, or shadow defects.

Do not judge translation meaning, target-language fluency, background repair, text placement, overflow, reading order, line breaking, or line spacing. Clear glyphs split across too many lines belong to Layout.

## Trigger Conditions

Use this Skill when any condition is present:

1. A candidate is `Major` or `Critical`.
2. The image has at least three `Minor` Text Rendering candidates.
3. Candidate confidence is below the configured threshold.
4. A candidate affects a high-risk region.
5. OCR and direct visual evidence disagree.
6. The base Agent cannot confidently distinguish blur, broken glyphs, residue, garbling, or overlap.

## Procedure

### Source-Target Delta Gate

Before confirming or updating any candidate, record `source_baseline`, `target_change`, and `translation_introduced` in `comparison`. Confirm the candidate only when the same localized source-target comparison shows that translation, erasure, or refill newly introduced the rendering defect or clearly worsened it.

Reject source-existing blur, crop, partial framing, edge proximity, faint styling, missing decoration, or capture damage when the target preserves it without clear deterioration. For `cropped_text`, require actual visible glyph loss newly present in the target. Text touching an edge, a perspective crop, or a photographed display already outside the source frame is not evidence of target cropping.

### Source-to-Target Completeness Pass

Compare every clearly visible source text role with its translated counterpart, including titles, labels, body text, buttons, step numbers, names, identifiers, model codes, dates, amounts, email addresses, warnings, and other key values. Confirm `missing_text` only when the source region is visually identifiable, the target has no translated or intentionally preserved replacement, and the content was not legitimately moved. Protected values may remain untranslated, but unexplained disappearance is still a visual completeness defect.

1. Begin with falsification: try to reject each candidate using source-preserved styling, intact glyph evidence, antialiasing, compression, moire, scan lines, or glare before confirming it.
2. Inspect only the candidate regions first; expand locally when context is needed.
2. Treat direct image evidence as primary and OCR as supporting evidence.
3. Confirm that the defect concerns rendered text rather than Layout or Background.
4. For size, weight, and color candidates, compare corresponding source and target regions and verify that they belong to the same visual hierarchy.
5. Apply the inactive-state preservation check: reject a mismatch when the source is already grey, faint, small, or de-emphasized for the same state or hierarchy role and the target preserves it.
6. For residue, verify that the alleged source characters form a continuous source fragment that is salient at normal viewing scale without bbox guidance; reject isolated letters, narrow edge slivers, zoom-only traces, and comparison-only traces.
7. Confirm, reject, or reclassify every candidate.
8. Calibrate severity from actual readability, affected scope, content importance, and user consequence.
9. Tighten or correct the translated-image `bbox_1000` when possible. Set `comparison.bbox_evidence_match` only when the box covers the text pixels named in evidence; reject or relocalize boxes on unrelated objects, neighboring text, background, or empty areas.
10. Add a new error only when it is visually explicit and independent.
12. Provide a short repair suggestion for each confirmed or new error.

For `source_text_residue`, return `comparison.residue_evidence.identifiable_source_strokes`, `matched_source_fragment`, `visible_at_normal_scale`, `salient_without_bbox_guidance`, `contiguous_source_fragment`, and `target_text_interference`. Reject it unless the first, third, fourth, and fifth fields are true and the matched source fragment is non-empty. Isolated letters or narrow background-text slivers that become noticeable only after bbox guidance are not errors and must not be downgraded to Minor.

For `broken_stroke`, return `comparison.glyph_damage_evidence.target_glyph_interruption_visible`, `capture_artifact_only`, `visible_at_normal_scale`, and `material_readability_impact`. Reject it when the target glyph is intact or the observation is only a capture artifact.

For `missing_glyph` or `garbled_text` suspected to come from unsupported target-font coverage, return `comparison.font_fallback_evidence.placeholder_glyphs_visible`, `visible_at_normal_scale`, `consecutive_placeholder_count`, `meaningful_text_region`, and `intended_text_recoverable`. Confirm `Critical` when at least three consecutive boxes, crossed boxes, tofu glyphs, or equivalent placeholders replace a meaningful line or phrase at normal scale and the intended visible text cannot be recovered. A single missing glyph or recoverable short run is not automatically Critical.

### Target-Only Recoverability Definition

Judge recoverability from the target image alone. The intended text is recoverable only when an ordinary viewer can identify the visible target content from the rendered target region itself. The Skill must not borrow the source wording, an expected translation, nearby semantic context, or external knowledge to reconstruct placeholders and then mark the target content recoverable.

When at least three consecutive placeholders are visible at normal scale, replace a meaningful line or phrase, and do not express recoverable target content, confirm `missing_glyph / Critical`. This gate does not require the rest of the page to be unusable. A single isolated unsupported symbol, a decorative mark, or a short run whose intended target content remains directly readable is not automatically Critical.

For `too_small_to_read`, complete `comparison.readability_evidence` with `source_readable_at_normal_scale`, `target_readable_at_normal_scale`, `translation_introduced_size_reduction`, and `unreadable_target_fragment`. Reject the candidate unless the source is readable at normal scale, the target is not, translation introduced the material size reduction, and a concrete unreadable target fragment or region can be named. Dense copy, source-preserved small print, or text that only looks small inside the review UI thumbnail is not an error.

For repeated typography symptoms, assign one `style_group_id` and set `shared_rendering_cause` true only when the regions share one rendering setting.

For font, weight, color, outline, glow, and shadow candidates, complete `comparison.style_loss_evidence` with `readability_impact`, `hierarchy_impact`, `functional_state_impact`, and `brand_identity_impact`. Missing decoration alone is insufficient for Major. Readable loss of an artistic font treatment is Minor at most unless direct source-target evidence names a material impact in one of those fields.

For `font_size_inconsistency`, complete `comparison.typography_major_evidence` with:

- `target_readable_at_normal_scale`: whether every affected target character remains readable without zoom;
- `characters_complete`: whether the affected target content has no clipped, hidden, or missing characters;
- `content_association_preserved`: whether the text still belongs unambiguously to the same item, control, or section;
- `overlap_or_clipping`: whether the size change newly causes actual glyph overlap or clipping;
- `functional_state_changed`: whether the size change makes a control state or actionable hierarchy materially misleading;
- `brand_identity_lost`: whether a primary brand or identity-bearing title becomes materially unrecognizable;
- `reading_requires_careful_inspection`: whether normal reading, rather than aesthetic comparison, requires extra effort.

Keep `font_size_inconsistency` as `Minor` when the target remains readable, characters are complete, content association is preserved, and there is no overlap, clipping, functional-state change, or brand-identity loss. A localized extra line, a split number or short phrase, abnormal emphasis, a larger bbox, `hierarchy_impact=material`, `disrupted_reading_flow`, or `rereading_required` does not independently justify `Major`.

Use `Major` only when at least one of the following concrete consequences is directly visible:

- the target text is not readable at normal scale;
- characters are clipped, hidden, or missing;
- actual glyph overlap or clipping is present;
- content association is broken;
- a functional state or actionable hierarchy becomes materially misleading;
- primary brand identity becomes materially unrecognizable.

`reading_requires_careful_inspection=true` does not independently justify `Major`; one of the concrete consequences above must also be present. If a required gate field is missing or uncertain, use `Minor`.

12. Return strict JSON only.

## Severity Refinement

Use `Minor` when the defect is slight and the text remains easy to read. Use `Major` when a meaningful region is visibly degraded or requires careful inspection. Use `Critical` only when primary or important text is unreadable, nearly unreadable, or unusable for a concrete high-risk action.

An unrecoverable meaningful line or phrase replaced by consecutive font-fallback placeholders is also `missing_glyph / Critical`, even outside a high-risk region, because the rendering failed to express that visible content at all.

For `missing_text`, use Minor for decorative or redundant content, Major for a meaningful but recoverable missing region, and Critical when primary content or a key identity, contact, warning, amount, navigation, or operation value disappears and makes the image unusable or materially untrustworthy.

Explicitly preserve Critical candidates for: unrecoverable runs of fallback boxes; loss of most labels in a functional keyboard or keypad; loss of core dates, results, identifiers, amounts, measurements, or protected values that makes a structured record materially untrustworthy; and large-scale garbling or omission that makes the primary text region unrecoverable. State the corresponding source content, target-only change, normal-scale visibility, and concrete unusable or untrustworthy consequence. The Visual Critic applies the final Critical Gate.

Do not retain a typography error when the translated text preserves the source's inactive-state, hierarchy, or emphasis styling. Source residue is Major only when unambiguous at normal viewing scale and materially disruptive; faint comparison-only residue is rejected.

Text overlap is not automatically `Critical`; judge the resulting readability and consequence. High-risk location alone is also insufficient for `Critical`.

Missing decoration alone is insufficient for Major. Do not upgrade a readable artistic-style difference merely because the original used a more elaborate typeface, outline, glow, shadow, or handwritten effect.

For font-size findings, do not use `hierarchy_impact=material`, a larger bbox, an extra line, a split number, or `rereading_required` alone to justify `Major`. Reapply `typography_major_evidence` after refinement and keep the result `Minor` unless a concrete Major gate is true.

## Suggestion Policy

Every confirmed or new error must include `suggestion`. Write it in `output_language`, identify the affected region or rendering property, and state one directly usable repair action. Do not rewrite translation semantics or give a long explanation.

## Output Contract

Return `updated_errors`, `new_errors`, `rejected_errors`, and `notes` exactly as defined in `references/output-schema.md`. Do not return scores or a Skill plan.

## References

Read only the reference needed for the current case:

- Error boundaries: `references/taxonomy.md`
- Severity decisions: `references/severity-rubric.md`
- Boundary examples: `references/examples.md`
- Output fields: `references/output-schema.md`

## Self-check

Before returning, verify every input candidate is updated, rejected, or intentionally preserved; each confirmed error has direct visual evidence, normalized `bbox_1000` information, confidence, and a suggestion; every `too_small_to_read` candidate has complete source-target `readability_evidence`; Layout and Background defects are excluded; and JSON contains no extra prose.
