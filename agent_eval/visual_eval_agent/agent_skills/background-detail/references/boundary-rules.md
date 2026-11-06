# Attribution Boundaries

- Damaged non-text pixels, textures, gradients, objects, icons, or decorations: Background.
- Blurred, malformed, cropped, residual, or stylistically mismatched text glyphs: Text Rendering.
- Clear text placed, aligned, wrapped, ordered, or rotated incorrectly: Layout.
- A translated block covers an intact object because of placement: Layout `content_occlusion`.
- Erasure physically removes or deforms the object beneath the source text: Background `scene_element_damage`.
- Readable original text remains: Text Rendering `source_text_residue`.
- Only a non-readable outline or shadow remains: Background `inpainting_residue`.
- Recognizable source glyphs plus their underline, glow, shadow, outline, or color streak: one Text Rendering `source_text_residue`; reject the Background symptom as a duplicate consequence.
- Retain a separate Background error only when damage extends beyond the source-text footprint into non-text pixels and requires an independent repair.

## Source-Text Footprint Fields

Every updated or new Background error must include `comparison.relation_to_source_text_footprint`, `comparison.recognizable_source_glyphs`, `comparison.independent_non_text_damage`, and `comparison.independent_repair_required`.

Compression, antialiasing, camera moire, display scan lines, and global exposure changes are not Background errors unless the target introduces a new localized discontinuity.
