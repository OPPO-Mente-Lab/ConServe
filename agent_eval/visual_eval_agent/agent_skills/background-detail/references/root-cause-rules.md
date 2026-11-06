# Root Cause Rules

One region and one causal repair failure produce one scored error.

- Recognizable source glyphs plus a same-footprint underline, glow, shadow, or color streak: retain Text Rendering `source_text_residue`; the Background observation is secondary evidence.
- If removing the residual source glyphs and their original decoration would also remove the alleged Background defect, the two observations share one root cause.
- Keep Background separate only when non-text damage remains after that counterfactual removal and needs an independent repair.

- Smear plus color change from the same patch: retain `smearing_artifact`.
- Texture break plus hard boundary from one inpaint mask: retain the more visually causal defect and record the other as impact.
- Damaged icon plus local edge residue caused by over-erasure: retain `scene_element_damage`.
- Missing source object plus blank color patch: retain `missing_scene_content`.
- Unsupported object plus false shadow: retain `unexpected_generated_content`.
- A source-existing tab, half-circle, button, panel, shadow, or decoration: reject `unexpected_generated_content`.

Keep separate errors only when defects are spatially distinct and require independent repairs. `root_cause_groups` records merged candidate IDs without adding penalties.

Spatial separation alone does not make errors independent. Merge repeated color blocks or smears within the same panel when they share one fill operation, visual treatment, and repair action. Never keep both a panel-wide summary error and the local patches summarized by it.

Repeated hard rectangular masks behind translated labels created by one shared fill operation are one `hard_inpainting_edge` or `color_discontinuity` root cause. List affected regions instead of creating one penalty per rectangle.
