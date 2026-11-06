# Output Schema

```json
{
  "decision": "confirmed_error|not_an_error|unresolved",
  "errors": [
    {
      "type": "omission",
      "severity": "Minor|Major|Critical",
      "source_block_ids": ["S01"],
      "source_fragment": "原图中的准确文字片段",
      "target_fragment": "",
      "source_bbox": [0, 0, 1000, 1000],
      "evidence": "简体中文证据",
      "suggestion": "'源文或错误译文'->'建议译文'",
      "confidence": 0.95
    }
  ],
  "excluded_source_block_ids": [
    {
      "block_id": "S02",
      "reason": "protected_text|already_covered|decorative|ocr_uncertain"
    }
  ],
  "notes": ["简体中文说明"]
}
```

`confirmed_error` 至少包含一个错误；`not_an_error` 和 `unresolved`
的 `errors` 必须为空。

