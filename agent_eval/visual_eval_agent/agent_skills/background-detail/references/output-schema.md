# Background Detail Output Schema

Return strict JSON with exactly this top-level shape:

```json
{
  "updated_errors": [
    {
      "original_error_id": "candidate id",
      "type": "allowed Background type",
      "severity": "Minor|Major|Critical",
      "region_id": "region id or null",
      "bbox_1000": {"affected_area": [0, 0, 0, 0]},
      "bbox_confidence": 0.0,
      "localization_status": "exact|approximate|uncertain|global",
      "impact_tags": [],
      "comparison": {
        "source_counterpart_present": false,
        "target_only_residue_visible": true,
        "relation_to_source_text_footprint": "inside|extends_beyond|separate|uncertain",
        "recognizable_source_glyphs": false,
        "independent_non_text_damage": false,
        "independent_repair_required": false,
        "capture_artifact_kind": "moire|scan_lines|glare|compression|none|uncertain",
        "capture_artifact_preservation": "preserved|weakened|missing|changed|uncertain",
        "independent_scene_content": null,
        "bbox_evidence_match": true,
        "unexpected_content_evidence": {
          "target_new_content": null,
          "target_new_content_visible": false,
          "source_counterpart_absent": false,
          "bbox_evidence_match": false
        },
        "texture_evidence": {
          "source_primitive": null,
          "source_primitive_visible": false,
          "target_break_visible": false,
          "visible_at_normal_scale": false
        },
        "scene_element_evidence": {
          "source_element": null,
          "source_element_visible": false,
          "target_element_missing_or_malformed": false,
          "functional_loss": false
        }
      },
      "severity_factors": {
        "visible_at_normal_scale": "true|false|uncertain",
        "requires_side_by_side_comparison": false,
        "readability_impact": "none|slight|material",
        "scene_element_impact": "none|slight|material|unrecoverable",
        "operation_impact": "none|material",
        "affected_scope": "small|medium|large",
        "rectangular_fill_visible": "true|false|uncertain",
        "repeated_rectangular_regions": 0,
        "contrast_with_surroundings": "low|medium|high|uncertain",
        "source_background_overwritten": "true|false|uncertain"
      },
      "evidence": "specific source-target visual diagnosis",
      "suggestion": "one actionable repair in output_language",
      "confidence": 0.0
    }
  ],
  "new_errors": [
    {
      "type": "allowed Background type",
      "severity": "Minor|Major|Critical",
      "region_id": "region id or null",
      "bbox_1000": {"affected_area": [0, 0, 0, 0]},
      "bbox_confidence": 0.0,
      "localization_status": "exact|approximate|uncertain|global",
      "impact_tags": [],
      "comparison": {
        "source_counterpart_present": false,
        "target_only_residue_visible": true,
        "relation_to_source_text_footprint": "inside|extends_beyond|separate|uncertain",
        "recognizable_source_glyphs": false,
        "independent_non_text_damage": false,
        "independent_repair_required": false,
        "capture_artifact_kind": "moire|scan_lines|glare|compression|none|uncertain",
        "capture_artifact_preservation": "preserved|weakened|missing|changed|uncertain",
        "independent_scene_content": null,
        "bbox_evidence_match": true,
        "unexpected_content_evidence": {
          "target_new_content": null,
          "target_new_content_visible": false,
          "source_counterpart_absent": false,
          "bbox_evidence_match": false
        },
        "texture_evidence": {
          "source_primitive": null,
          "source_primitive_visible": false,
          "target_break_visible": false,
          "visible_at_normal_scale": false
        },
        "scene_element_evidence": {
          "source_element": null,
          "source_element_visible": false,
          "target_element_missing_or_malformed": false,
          "functional_loss": false
        }
      },
      "severity_factors": {
        "visible_at_normal_scale": "true|false|uncertain",
        "requires_side_by_side_comparison": false,
        "readability_impact": "none|slight|material",
        "scene_element_impact": "none|slight|material|unrecoverable",
        "operation_impact": "none|material",
        "affected_scope": "small|medium|large",
        "rectangular_fill_visible": "true|false|uncertain",
        "repeated_rectangular_regions": 0,
        "contrast_with_surroundings": "low|medium|high|uncertain",
        "source_background_overwritten": "true|false|uncertain"
      },
      "evidence": "specific source-target visual diagnosis",
      "suggestion": "one actionable repair in output_language",
      "confidence": 0.0
    }
  ],
  "rejected_errors": [
    {
      "original_error_id": "candidate id",
      "reason": "source_artifact|compression_or_moire|text_rendering_issue|layout_issue|duplicate|insufficient_evidence",
      "evidence": "specific evidence contradicting the candidate",
      "confidence": 0.0,
      "direct_contradiction": true
    }
  ],
  "root_cause_groups": [
    {
      "primary_error_id": "retained candidate id",
      "merged_error_ids": ["duplicate candidate id"],
      "reason": "shared causal repair failure"
    }
  ],
  "notes": []
}
```

Every local error requires `affected_area`. For `global`, use `bbox_1000: null`. Every updated or new error requires a non-empty `suggestion`, complete `comparison`, and complete `severity_factors`. Do not return scores, penalties, image data, or prose outside JSON.
