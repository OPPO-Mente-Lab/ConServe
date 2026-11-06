# Layout Boundary Rules

## Layout versus Text Rendering

- Clear glyphs in the wrong position, direction, order, or container: Layout.
- Blurred, malformed, missing, garbled, residual, or stylistically mismatched glyph pixels: Text Rendering.
- Glyphs clipped because the text block leaves its container: Layout.
- Glyph pixels missing despite sufficient container space: Text Rendering.
- Tight line spacing without actual glyph-layer corruption: Layout.
- Two rendered text layers physically interfere with character recognition: Text Rendering.

## Layout versus Background

- Translated text covers an icon, person, product, or decoration: Layout.
- Source-text erasure damages that icon, person, product, texture, or decoration: Background.
- The translated block is correctly placed over a visibly smeared repair region: Background only.

## Layout versus text quality

Do not judge whether words are translated correctly, natural, concise, or suitable. Evaluate only their spatial presentation.

If another Agent reports the same pixels, preserve the Layout error only when spatial arrangement is the primary defect. Visual Critic performs final cross-dimension deduplication.
