# Background Detail Examples

## Slight color patch

The target contains a small, faint color mismatch in a plain wall; no edge or content is damaged.

Decision: `color_discontinuity / Minor`.

## Obvious gradient break

A smooth source sky gradient becomes a rectangular brightness step behind the translated text.

Decision: `gradient_discontinuity / Major`.

## Damaged decoration edge

The inpaint mask removes part of a meaningful emblem, but its identity remains recoverable.

Decision: `scene_element_damage / Major`.

## False generated object

The target introduces a new object-like shape that changes the interpretation of the scene.

Decision: `unexpected_generated_content / Critical` only when the changed interpretation is concrete; otherwise Major.

## Shared camera artifact

Both images contain identical scan lines and moire.

Decision: reject the candidate because the artifact was not introduced by translation repair.

## Source-Existing Pill Resized for Translation

The source contains a rounded solid-color system-message pill. The translated text is longer, so the target pill becomes wider while preserving its rounded boundary, contained fill, and visual role. It covers more of the same background pattern only because of this content-driven resizing.

Decision: reject the Background candidate. This is not smearing. Report only if color leaks outside the container, the rounded edge is visibly destroyed, or unrelated scene content is overwritten. Route an unreasonable size or placement change with an intact background to Layout.

## Source Text Decoration Remains

Recognizable source letters and their underline remain behind readable translated text. The underline appears as a colored streak, but it is confined to the original text footprint and disappears when the residual source text is removed.

Decision: reject the Background candidate as `text_rendering_issue`. Treat the letters, underline, and streak as one Text Rendering `source_text_residue / Minor` root cause.
