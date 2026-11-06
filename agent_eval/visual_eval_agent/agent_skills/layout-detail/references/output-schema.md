# Layout Detail Output Schema

Return strict JSON with exactly this top-level shape:

```json
{
  "updated_errors": [
    {
      "original_error_id": "candidate id",
      "type": "allowed Layout type",
      "severity": "Minor|Major|Critical",
      "region_id": "region id or null",
      "bbox_1000": {
        "affected_area": [0, 0, 0, 0],
        "text_area": [0, 0, 0, 0],
        "container_area": [0, 0, 0, 0],
        "occluded_area": [0, 0, 0, 0]
      },
      "bbox_confidence": 0.0,
      "localization_status": "exact|approximate|uncertain|global",
      "impact_tags": [],
      "comparison": {
        "bbox_evidence_match": true,
        "source_page_rotation": 0,
        "target_page_rotation": 0,
        "source_block_axis": "horizontal|vertical|rotated|mixed|uncertain",
        "target_block_axis": "horizontal|vertical|rotated|mixed|uncertain",
        "source_arrangement_preserved": false,
        "new_rotation_delta_degrees": 0,
        "new_head_rotation_required": false,
        "orientation_anchor_evidence": {
          "anchor_kind": "latin_text|number|punctuation|container_axis|ui_geometry|mixed|none",
          "source_anchor": "short visible anchor or null",
          "target_anchor": "matching visible anchor or null",
          "source_anchor_visible": true,
          "target_anchor_visible": true,
          "source_anchor_rotation_degrees": 0,
          "target_anchor_rotation_degrees": 0,
          "anchor_mismatch_confirmed": false,
          "cjk_glyph_shape_only": false
        },
        "line_break_evidence": {
          "scope": "single_word|short_phrase|multiple_breaks|passage",
          "characters_complete": true,
          "reading_order_preserved": true,
          "content_association_preserved": true,
          "overlap_or_clipping": false,
          "material_reconstruction_required": false
        }
      },
      "evidence": "specific visible diagnosis",
      "suggestion": "one actionable repair in output_language",
      "confidence": 0.0
    }
  ],
  "new_errors": [
    {
      "type": "allowed Layout type",
      "severity": "Minor|Major|Critical",
      "region_id": "region id or null",
      "bbox_1000": {
        "affected_area": [0, 0, 0, 0]
      },
      "bbox_confidence": 0.0,
      "localization_status": "exact|approximate|uncertain|global",
      "impact_tags": [],
      "comparison": {},
      "evidence": "specific visible diagnosis",
      "suggestion": "one actionable repair in output_language",
      "confidence": 0.0
    }
  ],
  "rejected_errors": [
    {
      "original_error_id": "candidate id",
      "reason": "short rejection reason",
      "evidence": "specific visual evidence that contradicts the candidate",
      "confidence": 0.0,
      "direct_contradiction": true
    }
  ],
  "root_cause_groups": [
    {
      "primary_error_id": "retained candidate id",
      "merged_error_ids": ["duplicate candidate id"],
      "reason": "shared causal defect"
    }
  ],
  "notes": []
}
```

`text_area`, `container_area`, and `occluded_area` are optional. Every local error requires `affected_area`. For `global`, use `bbox_1000: null`.

Every updated or new error requires a non-empty `suggestion`. Do not add scores or free-form prose outside JSON.
