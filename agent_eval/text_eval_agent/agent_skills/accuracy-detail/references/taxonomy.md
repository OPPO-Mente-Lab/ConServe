# Accuracy Detail Taxonomy

Use this reference when the candidate error type is unclear.

## Error Types

- `mistranslation`: The translated text expresses the wrong concept, action, object, relation, or user-facing meaning.
- `omission`: Source information is missing from the translation.
- `under_translation`: The translation weakens or underspecifies meaningful source information.
- `over_translation`: The translation adds specificity not justified by the source.
- `addition`: The translation adds extra information not present in the source.
- `hallucination`: The translation invents unsupported information that can mislead the user.
- `repetition`: The target contains unsupported machine-like loop repetition of a phrase, sentence, or paragraph that is absent from the source.
- `terminology_error`: Domain, technical, institutional, or UI terminology is translated inconsistently or incorrectly.
- `number_error`: Numbers, amounts, percentages, quantities, codes, or identifiers are wrong.
- `unit_error`: Units such as size, weight, currency, duration, or measurement are wrong.
- `date_error`: Dates, times, deadlines, or validity periods are wrong.
- `negation_error`: Negation, permission, prohibition, obligation, or conditional polarity is wrong.
- `relation_error`: Relationships between entities, conditions, causes, or actions are wrong.
- `text_pair_uncertain`: Source-target pairing is too uncertain for a strong faithfulness judgment.

## Boundary

If the target text is semantically correct but awkward, do not report an Accuracy error. Leave it to Fluency Style.

Use `repetition` only for a generation failure pattern. Do not use it for source-preserved repetition, deliberate rhetorical repetition, necessary grammatical repetition, repeated list labels, or one accidental duplicate word.
