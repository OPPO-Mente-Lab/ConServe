# Text Rendering Severity Rubric

Judge severity from outcome, not from the type name.

Use this order:

1. Determine how much translated text is affected.
2. Determine whether characters remain recognizable at normal viewing scale.
3. Determine whether the affected text is primary, secondary, decorative, or high-risk.
4. Determine whether the defect changes the user's ability to read or act.
5. If evidence is ambiguous, keep the lower severity and lower confidence.

## Minor

All should normally be true:

- The affected area is local.
- Characters remain easy to recognize.
- Reading speed and interpretation are essentially unchanged.
- The issue is cosmetic or only slightly reduces polish.

Examples include slight edge roughness, faint residue that does not touch glyphs, and a visible but cosmetic size, weight, or color mismatch whose text remains easy to read.

A suspected source trace visible only after zooming, bbox guidance, or repeated side-by-side comparison is not an error. A source-preserved inactive, unselected, or de-emphasized color is not an error.

Missing decoration alone is insufficient for Major. A readable loss of artistic font, outline, glow, shadow, weight, or handwritten character is Minor when information hierarchy, functional state, and brand identity remain materially intact.

## Major

At least one should be true:

- A meaningful text region requires careful inspection.
- Several characters are difficult to distinguish.
- Visible residue or text overlap interferes with local reading.
- Clipping, missing glyphs, or broken strokes remove part of readable content.
- A meaningful source text region or protected value is completely absent, but the image remains partly usable.
- Typography is conspicuously inconsistent and materially harms visual quality.
- Size, weight, or color changes destroy a meaningful hierarchy, require careful reading, or materially reduce contrast.

For source residue, Major additionally requires unambiguous source characters that are visible at normal viewing scale and interfere with local reading. For typography, Major requires a harmful change from the corresponding source role, not merely low contrast in the translated image.

For style candidates, name at least one material impact in `style_loss_evidence`: readability, hierarchy, functional state, or brand identity. Without one, cap the error at Minor.

The region remains at least partly usable. Major does not require a high-risk context.

## Critical

Use only when the rendering outcome is unusable or creates a concrete high-risk reading failure:

- Most primary translated text is unreadable or nearly unreadable.
- Extensive corruption prevents recognition of the intended visible text.
- An important warning, permission, safety instruction, amount, or other high-risk text cannot be read reliably.
- A semantic color change or lost emphasis makes critical status or warning information unreliable.
- Rendering defects make the translated image unusable for its principal text-reading purpose.
- Primary content or a key identity, contact, warning, amount, navigation, or operation value is completely missing and makes the image materially untrustworthy.
- At least three consecutive font-fallback boxes, crossed boxes, tofu glyphs, or equivalent placeholders replace a meaningful line or phrase at normal scale, and the intended visible text cannot be recovered.

## Non-automatic Escalations

- Text overlap is not automatically Critical.
- A large bbox is not automatically Critical.
- High-risk location is not automatically Critical.
- OCR failure is not automatically a rendering error.
- Several reports of the same pixels do not increase severity.
- One isolated missing glyph or a recoverable short placeholder run is not automatically Critical.

When deciding between Minor and Major, ask whether normal reading requires extra effort. When deciding between Major and Critical, ask whether the intended text can still be recovered reliably and whether failure has a concrete consequence.
