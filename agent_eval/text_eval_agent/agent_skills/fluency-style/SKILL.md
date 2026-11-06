---
name: fluency-style
description: Use when Fluency Agent errors are Major or Critical, repeated Minor, low-confidence, high-risk, unclear, ungrammatical, unnatural, stylistically mismatched, audience-mismatched, translationese, or garbled.
---

# Fluency Style Skill

## Purpose

Refine only `text.fluency` candidate errors.

This Skill checks whether translated text is natural, grammatical, readable, and appropriate as target-language text. It confirms, rejects, reclassifies, or adjusts severity for Fluency Agent candidate errors.

It also provides short actionable rewrite suggestions.

## Scope

Handle:

- grammar_error
- awkward_expression
- word_order_error
- collocation_error
- translationese
- unclear_expression
- style_mismatch
- audience_mismatch
- garbled_expression

Do not handle:

- mistranslation
- omission
- addition
- hallucination
- number_error
- unit_error
- date_error
- negation_error
- terminology accuracy
- visual rendering
- layout
- background repair
- punctuation width, capitalization, decorative symbols, markup, speaker tags, or metadata

If the target text is fluent but semantically wrong, reject the fluency error and leave it to Faithfulness.

Unsupported machine-like repetition belongs to Faithfulness as `repetition / Critical`. Reject Fluency candidates caused by the same repeated span unless an independent fluency defect remains after removing the repeated material.

## Trigger Conditions

Load this Skill when any of the following is true:

1. A Fluency Agent candidate error is `Major`.
2. A Fluency Agent candidate error is `Critical`.
3. The same text pair or same image contains at least 3 Minor fluency errors.
4. A candidate error has low confidence, for example `confidence < 0.65`.
5. The text is in a high-risk region and has any fluency concern.
6. The translated text appears garbled, unreadable, or difficult to interpret.

## Severity Refinement

Use `Minor` when:

- the text is understandable;
- the expression is slightly stiff or unnatural;
- the wording has light translationese;
- individual word choice or collocation is not ideal;
- punctuation or spacing issue does not affect reading.

Every identifiable light naturalness defect is `Minor`, including a clearly local stiff phrase, mild translationese, light redundancy, or imperfect collocation whose meaning remains immediately understandable. Formatting and metadata are not scored and must be rejected rather than refined.

Use `Major` when:

- grammar is clearly wrong;
- word order noticeably disrupts reading;
- collocation is clearly unnatural;
- the reader needs to infer meaning;
- expression is understandable but visibly poor;
- style or register clearly does not fit the scene or expected audience.

Apply the Major readability gate before confirming `Major`:

- the reader must reconstruct the sentence structure, participant roles, action, or logical relation; or
- multiple interacting defects prevent a direct reading of the sentence.

A single local collocation, literal phrase, or non-native word choice is `Minor` when the intended meaning remains immediately clear and the sentence remains structurally usable.

Use `Critical` when:

- the text is almost unreadable;
- the text contains large garbled or corrupted fragments;
- word order or expression causes serious misunderstanding;
- users cannot reliably understand important instructions, warnings, guidance, permission text, safety text, legal text, medical text, or public information.

High-risk context does not automatically mean Critical. Upgrade only when the fluency issue materially blocks or distorts understanding.

## Procedure

1. Read `translated_text` as target-language text.
2. Use `source_text` only as light context, not for faithfulness scoring.
3. Review candidate errors from Fluency Agent.
4. Decide whether each candidate error should be confirmed, rejected, reclassified, or severity-adjusted.
5. Check whether the issue is actually a Faithfulness error. If yes, reject or downgrade the fluency error.
6. If the issue is confirmed, write a short concrete `suggestion`.
7. Add new fluency errors only when they are explicit and important.
8. Return JSON only.

For a Cross-region bilingual-collapse candidate, confirm one `awkward_expression / Minor` only when the batch evidence shows two distinct semantically equivalent source regions, one source region is already in the target language, and translation of the other creates duplicate target-language copy in separate visible regions. Keep the error on the newly translated pair, never reclassify it as machine `repetition`, and suggest removing the redundant translated rendering or preserving a deliberate bilingual strategy. Reject the candidate when it relies only on surface similarity.

## Suggestion Policy

Every confirmed or new error must include `suggestion`.

`suggestion` must use the exact arrow format `'<left>'->'<right>'` with English single quotes and no spaces around `->`:

- `left`: non-empty `source_fragment` when available; otherwise the incorrect `target_fragment`
- `right`: a natural target-language rewrite for that span
- deletion: set `right` to `[删除]`

Do not write free-form instructions such as `Change X to Y`. Keep the rewrite short, usable, and faithful to source context. Do not optimize for compact interface wording unless the input explicitly says the text is compact interface copy.

When `output_language` is `zh-CN`, write `evidence`, rejection reasons, and `notes` in Simplified Chinese. Keep JSON keys, type values, severity values, exact source/target fragments, and the `suggestion` arrow format unchanged.

## Output Contract

Return JSON only.

```json
{
  "updated_errors": [
    {
      "original_error_id": "string",
      "type": "grammar_error|awkward_expression|word_order_error|collocation_error|translationese|unclear_expression|style_mismatch|audience_mismatch|garbled_expression",
      "severity": "Minor|Major|Critical",
      "target_fragment": "target-side text span",
      "evidence": "specific fluency reason",
      "suggestion": "'source or bad target'->'suggested rewrite'",
      "confidence": 0.0
    }
  ],
  "new_errors": [
    {
      "type": "grammar_error|awkward_expression|word_order_error|collocation_error|translationese|unclear_expression|style_mismatch|audience_mismatch|garbled_expression",
      "severity": "Minor|Major|Critical",
      "target_fragment": "target-side text span",
      "evidence": "specific fluency reason",
      "suggestion": "'source or bad target'->'suggested rewrite'",
      "confidence": 0.0
    }
  ],
  "rejected_errors": [
    {
      "original_error_id": "string",
      "reason": "why this is not a fluency error"
    }
  ],
  "notes": []
}
```

## References

Read only the reference needed for the current case:

- For fluency/style categories, see `references/style-rubric.md`.
- For edge cases and few-shot patterns, see `references/examples.md`.
- For output contract details, see `references/output-schema.md`.

## Self-check

Before returning, verify:

- output is valid JSON only;
- every confirmed error is a target-language fluency issue;
- semantic accuracy issues are not duplicated as fluency errors;
- every confirmed or new error has `target_fragment`, `evidence`, `suggestion`, and `confidence`;
- `suggestion` matches `'<left>'->'<right>'` and the right side is the target-language rewrite or `[删除]`;
- severity follows Minor / Major / Critical boundaries.
