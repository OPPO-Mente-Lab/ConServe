# Text Rendering Detail Output Schema

Return JSON only. Every confirmed update or new error requires `suggestion`.

```json
{
  "updated_errors": [
    {
      "original_error_id": "string",
      "type": "allowed Text Rendering type",
      "severity": "Minor|Major|Critical",
      "region_id": "string or null",
      "bbox_1000": {
        "affected_area": [0, 0, 0, 0],
        "text_area": [0, 0, 0, 0]
      },
      "bbox_confidence": 0.0,
      "localization_status": "exact|approximate|uncertain|global",
      "comparison": {
        "bbox_evidence_match": true,
        "residue_evidence": {
          "identifiable_source_strokes": false,
          "matched_source_fragment": null,
          "visible_at_normal_scale": false,
          "salient_without_bbox_guidance": false,
          "contiguous_source_fragment": false,
          "target_text_interference": "none|slight|material"
        },
        "glyph_damage_evidence": {
          "target_glyph_interruption_visible": false,
          "capture_artifact_only": false,
          "visible_at_normal_scale": false,
          "material_readability_impact": false
        },
        "font_fallback_evidence": {
          "placeholder_glyphs_visible": false,
          "visible_at_normal_scale": false,
          "consecutive_placeholder_count": 0,
          "meaningful_text_region": false,
          "intended_text_recoverable": true
        },
        "readability_evidence": {
          "source_readable_at_normal_scale": true,
          "target_readable_at_normal_scale": false,
          "translation_introduced_size_reduction": true,
          "unreadable_target_fragment": "short fragment or region label"
        },
        "style_loss_evidence": {
          "readability_impact": "none|slight|material",
          "hierarchy_impact": "none|slight|material",
          "functional_state_impact": "none|material",
          "brand_identity_impact": "none|material"
        },
        "style_group_id": null,
        "shared_rendering_cause": false
      },
      "evidence": "specific visible reason",
      "suggestion": "short actionable rendering repair",
      "confidence": 0.0
    }
  ],
  "new_errors": [
    {
      "type": "allowed Text Rendering type",
      "severity": "Minor|Major|Critical",
      "region_id": "string or null",
      "bbox_1000": null,
      "bbox_confidence": 0.0,
      "localization_status": "global",
      "comparison": {},
      "evidence": "specific visible reason",
      "suggestion": "short actionable rendering repair",
      "confidence": 0.0
    }
  ],
  "rejected_errors": [
    {
      "original_error_id": "string",
      "reason": "not_visible|layout_issue|background_issue|duplicate|insufficient_evidence",
      "evidence": "specific visual evidence that contradicts the candidate",
      "confidence": 0.0,
      "direct_contradiction": true
    }
  ],
  "notes": []
}
```

Every bbox uses normalized integers in `[ymin, xmin, ymax, xmax]` order. `text_area` is optional. Include only the comparison object relevant to the error type. Use `bbox_1000: null` only with `localization_status: global`. Do not output scores, penalties, credentials, image data, or prose outside the JSON object.
