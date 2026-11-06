# Fluency Style Output Schema

Return JSON only.

Every confirmed or new error must include `suggestion`.

`suggestion` must be exactly `'<left>'->'<right>'`:

- `left` = non-empty `source_fragment`, else incorrect `target_fragment`
- `right` = natural target-language rewrite, or `[删除]` for deletions

```json
{
  "updated_errors": [
    {
      "original_error_id": "string",
      "type": "grammar_error|awkward_expression|word_order_error|collocation_error|translationese|unclear_expression|style_mismatch|audience_mismatch|garbled_expression",
      "severity": "Minor|Major|Critical",
      "target_fragment": "string",
      "evidence": "string",
      "suggestion": "'source or bad target'->'suggested rewrite'",
      "confidence": 0.0
    }
  ],
  "new_errors": [
    {
      "type": "grammar_error|awkward_expression|word_order_error|collocation_error|translationese|unclear_expression|style_mismatch|audience_mismatch|garbled_expression",
      "severity": "Minor|Major|Critical",
      "target_fragment": "string",
      "evidence": "string",
      "suggestion": "'source or bad target'->'suggested rewrite'",
      "confidence": 0.0
    }
  ],
  "rejected_errors": [
    {
      "original_error_id": "string",
      "reason": "string"
    }
  ],
  "notes": []
}
```
