# Accuracy Detail Examples

## Mistranslation

Source: `Keep door closed`

Translation: `保持窗户关闭`

Output: `mistranslation / Major`

Evidence should explain that `door` was transferred as `窗户`, changing the object of the instruction.

## Omission

Source: `Bring photo ID`

Translation: `请携带照片`

Output: `omission / Major`

Evidence should explain that `ID` is missing, so the required document changes.

## Under Translation

Source: `Wash hands thoroughly`

Translation: `洗手`

Output: `under_translation / Minor`

Evidence should explain that `thoroughly` is weakened or omitted while the core instruction remains understandable.

## Over Translation

Source: `Use protective equipment`

Translation: `必须使用护目镜`

Output: `over_translation / Major`

Evidence should explain that the target narrows general protective equipment into a specific item not stated by the source.

## Addition

Source: `Meeting starts at 9:00`

Translation: `会议9点开始，请提前30分钟到达`

Output: `addition / Major`

Evidence should explain that the instruction to arrive 30 minutes early is added without source support.

## Hallucination

Source: `Visitor registration`

Translation: `访客登记，需进行人脸识别`

Output: `hallucination / Major`

Evidence should explain that face recognition is invented and not supported by the source.

## Number Error

Source: `Room 204`

Translation: `2040房间`

Output: `number_error / Major`

Evidence should explain that the room number changed from `204` to `2040`.

## Unit Error

Source: `Wait 5 minutes`

Translation: `等待5小时`

Output: `unit_error / Major`

Evidence should explain that the duration unit changed from minutes to hours.

## Date Or Time Error

Source: `Arrives at 18:30`

Translation: `18:00到达`

Output: `date_error / Major`

Evidence should explain that the arrival time changed from `18:30` to `18:00`.

## Negation Error

Source: `Do not enter`

Translation: `可以进入`

Output: `negation_error / Critical`

Evidence should explain that the prohibition is reversed into permission and can affect safety or access decisions.

## Permission Error

Source: `Allow camera access`

Translation: `拒绝相机访问`

Output: `negation_error / Critical`

Evidence should explain that permission is reversed into refusal and may affect user operation or privacy control.

## Relation Error

Source: `Children must be accompanied by adults`

Translation: `成人必须由儿童陪同`

Output: `relation_error / Major`

Evidence should explain that the accompaniment relationship is reversed.

## Terminology Error

Source: `Emergency exit`

Translation: `紧急入口`

Output: `terminology_error / Critical`

Evidence should explain that `exit` is rendered as `入口`, which can mislead evacuation behavior.

## Medical Or Safety Amount Error

Source: `Take 1 tablet daily`

Translation: `每天服用2片`

Output: `number_error / Critical`

Evidence should explain that the dosage changed from `1` to `2`, which can affect health-related decisions.

## Text Pair Uncertain

Source: `Gate B12`

Translation: `请出示证件`

Output: `text_pair_uncertain / Major`

Evidence should explain that the source and target appear to describe unrelated information, so the pair should not be judged as a direct translation.

## Boundary With Fluency

Source: `Try again`

Translation: `再试一次`

Output: no Accuracy error. The meaning is preserved.
