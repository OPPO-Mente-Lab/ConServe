# Fluency Style Examples

## Minor: Slightly Literal

Source: `Try again`

Translation: `再次进行尝试`

Output: `awkward_expression / Minor`

Evidence: The expression is stiff but understandable.

Suggestion: `改为“再试一次”。`

## Minor: Light Translationese

Source: `Children must be accompanied by an adult`

Translation: `儿童必须被一个成年人陪伴着`

Output: `translationese / Minor`

Evidence: The expression has source-language structure but remains clear.

Suggestion: `改为“儿童须由成人陪同”。`

## Major: Collocation Error

Source: `Wet floor`

Translation: `地板正在湿`

Output: `collocation_error / Major`

Evidence: The collocation is unnatural and the reader must infer the intended warning.

Suggestion: `改为“地面湿滑”。`

## Major: Word Order Problem

Source: `Please keep quiet in the reading room`

Translation: `请在阅读房间进行安静的保持`

Output: `word_order_error / Major`

Evidence: Word order and phrasing are visibly unnatural.

Suggestion: `改为“请在阅览室保持安静”。`

## Critical: Garbled Expression

Translation: `请将进行的门关闭它`

Output: `garbled_expression / Critical`

Evidence: The text is structurally broken and cannot be read as natural target-language text.

Suggestion: `改为“请关门”。`

## Boundary With Faithfulness

Source: `Do not enter`

Translation: `可以进入`

Output: reject the Fluency error if the Chinese is fluent. The main issue is Faithfulness because the meaning is reversed.
