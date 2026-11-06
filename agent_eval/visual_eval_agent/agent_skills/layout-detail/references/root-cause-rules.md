# Root Cause Rules

One affected region and one causal defect produce one scored error.

## Causal chain

When a translated block is shifted, crosses a boundary, and covers an icon:

- primary type: `position_shift`;
- impact tags: `text_overflow`, `content_occlusion`;
- scored errors: one.

When unreasonable line breaking creates crowding in the same block:

- primary type: `unreasonable_line_break` if line division is the direct visible cause;
- impact tag: `disrupted_reading_flow`;
- do not add `text_region_crowding` unless crowding is independent.

When wrong-region placement changes which speaker owns dialogue:

- primary type: `wrong_region_fill`;
- impact tag: `wrong_content_association`;
- severity may be Critical when attribution is materially changed.

When repeated message cards or `Read more...` links share one alignment defect:

- retain one Layout error with all affected regions named;
- do not score every repeated instance separately.

When left alignment, fixed container width, and an awkward line split come from one text-block format setting:

- retain one dominant Layout error;
- treat the other symptoms as evidence or impact.

When multiple text regions collapse into one target area:

- retain one `text_region_crowding` root cause;
- treat unreadable overlap as a secondary Text Rendering consequence.

## Separate errors

Keep errors separate when they affect different containers or have independent causes. A direction defect and an unrelated position shift in the same bubble may remain separate because correcting one does not correct the other.

## Uncertain cause

Do not invent a hidden generation cause. Choose the most specific directly observable type, lower confidence, and explain the uncertainty briefly.

## Root-cause groups

Use `root_cause_groups` to declare which candidate IDs were merged. The first ID is the retained primary error. Grouping never creates another scored error.
