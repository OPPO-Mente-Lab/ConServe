# Accuracy Detail Output Schema

Return JSON only.

Every confirmed or new error must include `suggestion`.

`suggestion` must be exactly `'<left>'->'<right>'`:

- `left` = non-empty `source_fragment`, else incorrect `target_fragment`
- `right` = concrete suggested rewrite in the target language
- deletion uses `right = [删除]`

When `output_language` is `zh-CN`, diagnostic prose (`evidence`, rejection `reason`, and `notes`) must be Simplified Chinese. Exact source/target fragments and the arrow-format `suggestion` keep their structured form.

```json
{
  "updated_errors": [
    {
      "original_error_id": "string",
      "type": "string",
      "severity": "Minor|Major|Critical",
      "source_fragment": "string",
      "target_fragment": "string",
      "evidence": "string",
      "suggestion": "'source or bad target'->'suggested rewrite'",
      "confidence": 0.0
    }
  ],
  "new_errors": [
    {
      "type": "string",
      "severity": "Minor|Major|Critical",
      "source_fragment": "string",
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
