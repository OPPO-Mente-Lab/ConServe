---
name: accuracy-detail
description: Refines Faithfulness Agent candidate errors for image translation text evaluation. Use when faithfulness errors are Major or Critical, low-confidence, high-risk, or involve numbers, units, dates, negation, permission, safety, legal, medical, terminology, omission, addition, hallucination, repetition, or source-target pairing uncertainty.
---

# Accuracy Detail Skill

## Purpose

Refine only `text.faithfulness` candidate errors. Confirm, reject, reclassify, or update severity for source-to-target information transfer issues.

## Scope

Handle:

- mistranslation
- omission
- under_translation
- over_translation
- addition
- hallucination
- repetition
- terminology_error
- number_error
- unit_error
- date_error
- negation_error
- relation_error
- text_pair_uncertain

Do not handle fluency, UI style, visual rendering, layout, or background issues.

## Procedure

1. Check whether the source and translated text pair is reliable enough to judge.
2. Compare source meaning units against translated meaning units.
3. Apply the Contextual translatability gate to unchanged source-language fragments.
   - Wayfinding signs, street names, area names, venue names, and category labels normally require translation or transliteration.
   - Require affirmative preservation evidence before treating unchanged text as a valid brand, registered mark, model, URL, code, or conventional official name.
   - Capitalization, title case, stylization, or sign placement alone is not affirmative preservation evidence.
   - Confirm `under_translation / Major` when a complete user-facing label remains unchanged without that evidence.
   - Confirm `under_translation / Major` when a complete sentence or paragraph remains in the source language. Ignore minor OCR spacing or capitalization differences when the readable content is substantially the same.
   - Upgrade only when most primary content is untranslated and the main information task is unusable.
4. Inspect high-risk facts first: amounts, identifiers, safety, privacy, permissions, legal obligations, medical guidance, dates, units, negation, conditions, and operation-critical instructions.
5. Decide whether each candidate error should be confirmed, rejected, reclassified, or severity-adjusted.
   - When an existing source token is misread because of case, acronym, abbreviation, or entity ambiguity, confirm it as `mistranslation`, not `addition`. Preserve the exact source token and incorrect target token in the fragments.
6. Add new faithfulness errors only when the evidence is explicit.
7. For every confirmed or new error, write a short repair suggestion.
8. Return JSON only.

Every confirmed or new error must include `suggestion`.

Confirm `repetition / Critical` when the target contains unsupported machine-like loop repetition of a phrase, sentence, or paragraph that is absent from the source and makes the generated translation unreliable. Reject this label for repetition already present in the source, deliberate rhetorical repetition, repeated list labels, necessary grammatical repetition, or a single local duplicate-word typo.

`suggestion` must use the exact arrow format `'<left>'->'<right>'` with English single quotes and no spaces around `->`:

- `left`: non-empty `source_fragment` when available; otherwise the incorrect `target_fragment`
- `right`: the concrete suggested target-language rewrite for that span
- deletion: set `right` to `[删除]`, e.g. `'top laner '->'[删除]'`

Do not write free-form instructions such as `Change X to Y` or `将X修改为Y`. Do not repeat the evidence as the suggestion.

For an unchanged user-facing sign label, put the source label on the left and a reasonable translated/transliterated candidate on the right. If the official localized name cannot be verified, still provide a candidate rewrite rather than inventing official status in prose.

The input field `output_language` controls diagnostic prose. When it is `zh-CN`, write `evidence`, rejection reasons, and `notes` in Simplified Chinese. Keep JSON keys, type values, severity values, source/target fragments, and the `suggestion` arrow format unchanged.

## Output

Return:

```json
{
  "updated_errors": [
    {
      "original_error_id": "id",
      "type": "error_type",
      "severity": "Minor|Major|Critical",
      "source_fragment": "source span",
      "target_fragment": "target span",
      "evidence": "concrete reason and evidence",
      "suggestion": "'source or bad target'->'suggested rewrite'",
      "confidence": 0.0
    }
  ],
  "new_errors": [],
  "rejected_errors": [],
  "notes": []
}
```

## References

Read only the reference needed for the current case:

- For error categories, see `references/taxonomy.md`.
- For severity decisions, see `references/severity-rubric.md`.
- For edge cases and few-shot patterns, see `references/examples.md`.
- For output contract details, see `references/output-schema.md`.
