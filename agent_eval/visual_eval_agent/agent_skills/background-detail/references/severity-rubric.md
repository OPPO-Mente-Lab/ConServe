# Background Severity Rubric

## Imperceptible Surface Trace Gate

Do not report a surface-repair Minor when the trace is invisible at normal fit-to-screen scale, requires zooming or repeated source-target toggling, has low contrast and small scope, and has no readability, scene, operation, or independent non-text impact. Very faint haze, tiny dots, weak tonal variation, and suspected erasure marks below this threshold are no error.

A Minor surface defect must be noticeable at normal scale or readily apparent in one direct side-by-side comparison.

## Minor

Use Minor when the defect is small or cosmetic but noticeable at normal scale or readily apparent in one direct side-by-side comparison, and it does not materially affect reading, operation, a meaningful scene element, or a large area. Typical cases are a visible halo, slight color shift, short texture break, small ghost, or mildly unnatural patch that passes the Imperceptible Surface Trace Gate.

For a texture claim, `texture_evidence.source_primitive` must identify a concrete source pattern or edge. Generic claims such as "smoother texture" or "missing detail" are Minor at most.

## Major

For surface repair errors, both must hold: the defect is clearly visible at normal viewing scale without side-by-side comparison, and it has material readability, operation, meaningful-element, or large-area impact. A visible color or pixel difference alone is not Major. Scene-element damage, missing content, and unexpected generated content are judged from their recognizable visual or semantic consequence.

A new high-contrast rectangular refill is also Major when it is obvious at normal scale, its straight-edged solid block strongly contrasts with the immediately adjacent background, and comparison confirms that it overwrites a continuous source background or non-text scene area. Keep this exception narrow: a mild color mismatch, illumination change, glare, compression artifact, source-existing rectangle, or zoom-only patch remains Minor.

Repeated medium-contrast rectangular refills may also be one Major root cause when at least two normal-scale-visible blocks share one fill operation, overwrite continuous source background, and cumulatively damage a medium or large portion of a meaningful panel.

For scene-element damage, `scene_element_evidence.target_element_missing_or_malformed` must be true. If the element remains visible and functional, reject the candidate.

For `unexpected_generated_content`, the target element must be visible, the source counterpart must be confirmed absent, and the bbox must cover the named element. Reject source-existing panels, tabs, half-circles, buttons, shadows, decorations, and controls.

## Critical

Require a concrete critical impact: substantial primary-subject destruction; a key icon removed or unrecognizable; large-area corruption makes the image unusable; generated content changes scene interpretation; or safety, payment, or price information becomes unreadable or untrustworthy because its background is damaged.

Do not infer Critical from area, risk category, or error type alone.
