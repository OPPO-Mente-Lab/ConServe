# Output Schema

Return one JSON object and no surrounding prose.

Each block requires `block_id`, `text`, `bbox_1000`, `language`, `reading_order`, `block_type`, `visibility`, and `confidence`.

Each evaluation unit requires `unit_id`, `unit_type`, `source_sequence`, `target_sequence`, `alignment_type`, `pair_confidence`, and `warnings`.

Every sequence item requires an existing `block_id` and `join_after` from `none`, `space`, `newline`, or `paragraph`.

Every block must appear at most once across evaluation units and the unmatched array on its own side. Coordinates must use `[ymin, xmin, ymax, xmax]` integers in the inclusive 0 to 1000 range, with minimum values smaller than maximum values.
