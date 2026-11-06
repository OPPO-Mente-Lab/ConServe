# Accuracy Detail Severity Rubric

Judge severity by user impact, not by how awkward the wording looks.

Use this order:

1. Decide whether the source meaning is preserved.
2. Decide whether a core fact, action, object, condition, relation, number, unit, or time changed.
3. Decide whether the changed meaning can affect operation, access, safety, privacy, legal obligation, medical guidance, identity verification, or another important decision.
4. If evidence is weak or pairing is unreliable, lower confidence instead of upgrading severity.

## Minor

Use Minor when the translation is still usable and the core information remains correct.

Minor checklist:

- The main action, object, condition, and outcome are preserved.
- The error affects nuance, intensity, completeness, or secondary detail.
- A reasonable user would still make the same decision or perform the same action.
- The issue does not alter a number, unit, date, permission, prohibition, obligation, identity, or safety-critical instruction.

Typical Minor cases:

- A modifier is weakened: `thoroughly` becomes implicit.
- A descriptive detail is less precise but still directionally correct.
- A general instruction remains understandable despite losing emphasis.
- A low-impact label is slightly underspecified.
- Redundant restatement of the same meaning that does not introduce a new proposition, action, condition, or entity.
- A conventional term is imperfect but the referent and operation remain clear.
- A small but identifiable semantic shift changes intensity, nuance, specificity, or breadth while preserving the core proposition, action, participants, and outcome.
- A small nonessential qualifier on a user-facing label remains untranslated, while the translated main label still supports identification and navigation.

## Major

Use Major when meaning clearly changes but the error is not directly high-risk or decision-critical.

Major checklist:

- A core concept, action, object, condition, relation, number, unit, date, or named entity is wrong or missing.
- A core action, participant role, relation, or named entity changes, even when the resulting sentence remains readable.
- The user may misunderstand the content.
- The text remains interpretable, but it no longer faithfully represents the source.
- The issue does not directly change safety, permission, access, legal, medical, identity, or operation-critical decisions.

Typical Major cases:

- Object changed: `door` becomes `window`.
- Time changed in an informational schedule.
- A required condition is omitted from a low-risk instruction.
- A relationship between two entities is reversed in non-critical context.
- Unsupported specificity is added and changes the meaning.
- An addition introduces a substantively new proposition, action, condition, participant, or outcome.
- A complete wayfinding, street, area, venue, or category label remains unchanged in the source language without affirmative preservation evidence. Treat this as `under_translation / Major`.
- A complete sentence or paragraph remains in the source language while other primary content is translated. Treat this as `under_translation / Major`.

## Critical

Use Critical only when the error can make the translated text unsafe, unusable, or materially misleading for an important decision.

Critical checklist:

- The error affects safety, medical guidance, legal obligation, privacy, permission, access control, identity verification, security, or operation-critical instructions.
- Negation, prohibition, permission, obligation, or condition is reversed.
- A dosage, required amount, threshold, deadline, location, access state, or emergency instruction is materially wrong.
- The user could perform the wrong action, lose access, accept the wrong obligation, expose private data, or face safety risk.

Typical Critical cases:

- `Do not enter` becomes `You may enter`.
- `Allow camera access` becomes `Deny camera access`.
- `Take 1 tablet daily` becomes `Take 2 tablets daily`.
- `Turn off power before repair` loses `Turn off power`.
- A legal or consent statement reverses allow/deny, must/may, or agree/decline.
- The target enters unsupported machine-like repetition of a phrase, sentence, or paragraph. This `repetition` failure is Critical even in normal-risk text.

## Upgrade to Critical when

- The same error would be Major in ordinary text but appears in a high-risk region.
- A number, unit, date, deadline, or condition controls access, safety, medical use, identity verification, legal obligation, or an operation.
- The translation changes what the user is allowed, forbidden, required, or warned to do.
- The target text would likely cause a wrong real-world action.
- A confirmed unsupported loop-like `repetition` failure makes the generated translation unreliable.

## Keep as Major when

- The meaning is clearly wrong, but the text is informational and not tied to an important decision.
- The affected number, date, object, or relationship is not safety-, access-, legal-, medical-, privacy-, or operation-critical.
- The user may misunderstand content, but the consequence is limited to comprehension quality.

## Do not upgrade when

- The only problem is awkward target-language expression.
- The issue is a mild nuance loss and the core instruction is preserved.
- The source-target pair may be mismatched or OCR is unreliable.
- The high-risk label is present but the actual error does not affect the high-risk meaning.

## Unchanged names and signs

- Capitalization, title case, stylization, or sign placement alone does not prove that text must remain unchanged.
- Accept unchanged text only with affirmative preservation evidence: a visible brand or registered mark, a model/code/URL, or reliable context establishing an official name conventionally kept unchanged.
- When a whole user-facing sign label is unchanged and no such evidence exists, confirm `under_translation / Major`.
- When only a secondary qualifier is unchanged and users can still identify the destination or function, use Minor.
- When a complete sentence or paragraph remains in the source language, use `under_translation / Major`; use Critical only when most primary content is untranslated and the main information task is unusable.
- The repair suggestion should translate or transliterate the label and flag official-name verification when needed.

## Tie-breakers

- If deciding between Minor and Major, identify the minimal correction first. If it only restores nuance, removes equivalent repetition, or replaces an imperfect but recognizable conventional term, use Minor.
- If the minimal correction restores a core action, participant role, subject-object relation, condition, named entity, or distinct proposition, use Major.
- If deciding between Major and Critical, ask: could this change a user action, right, access, safety, privacy, medical, legal, or security decision? If yes, use Critical.
- If evidence is insufficient, keep the lower severity and lower confidence.
- Do not downgrade confirmed `repetition` because the region is normal risk; instead verify carefully that it is machine-like and unsupported by the source.
- If multiple small omissions combine to change core meaning, treat the combined issue as Major.
- If a candidate error is really target-language awkwardness with preserved meaning, reject it as Accuracy and leave it to Fluency.

## Uncertainty

If the source-target pair is unreliable or evidence is insufficient, do not upgrade based only on suspicion. Use `text_pair_uncertain` or lower confidence.
