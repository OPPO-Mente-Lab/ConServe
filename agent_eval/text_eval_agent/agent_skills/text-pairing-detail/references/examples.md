# Structural Examples

## Paragraph split across lines

Two English lines inside one paragraph container may form one unit with `space`. Two Chinese lines that split a continuous sentence may use `none` when no punctuation or spacing belongs between them.

## Comic speech bubbles

Lines inside one bubble may form one unit in the bubble's original reading direction. Text from different bubbles remains in separate units even when the bubbles are close.

## Table

Text in adjacent cells remains separate. A source cell translated into two visual lines in the corresponding target cell is a one-to-many unit, not two unrelated units.

## Uncertain alignment

When a clear source label has no reliable translated counterpart, keep its ID in `unmatched_source_block_ids`. Do not align it to the nearest target paragraph merely because the boxes overlap.
