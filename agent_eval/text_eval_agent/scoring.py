# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import Counter

from severity_scoring import score_severity_counts

from .config import TextEvaluationConfig
from .models import TextDimension, TextError


def _dimension_result(
    errors: list[TextError],
    dimension: TextDimension,
) -> dict[str, float | int]:
    counts = Counter(
        error.severity.value
        for error in errors
        if error.dimension == dimension
    )
    score_5 = score_severity_counts(
        counts["Minor"],
        counts["Major"],
        counts["Critical"],
    )
    return {
        "minor_count": counts["Minor"],
        "major_count": counts["Major"],
        "critical_count": counts["Critical"],
        "score_5": score_5,
        "score_100": round(score_5 * 20, 4),
    }


def score_text_errors(
    errors: list[TextError],
    config: TextEvaluationConfig,
    caps: list[dict] | None = None,
) -> dict:
    # `caps` remains in the public signature for old callers. The shared
    # severity formula now handles Critical errors directly.
    del caps
    faithfulness = _dimension_result(errors, TextDimension.FAITHFULNESS)
    fluency = _dimension_result(errors, TextDimension.FLUENCY)
    critical_count = faithfulness["critical_count"] + fluency["critical_count"]

    if critical_count:
        text_score_5 = 0.0
        scoring_method = "weighted_dimension_average_with_critical_cap"
    else:
        text_score_5 = round(
            config.faithfulness_weight * faithfulness["score_5"]
            + config.fluency_weight * fluency["score_5"],
            4,
        )
        scoring_method = "weighted_dimension_average"

    text_score_100 = round(text_score_5 * 20, 4)
    return {
        "faithfulness": faithfulness,
        "fluency": fluency,
        "dimension_weights": {
            "faithfulness": config.faithfulness_weight,
            "fluency": config.fluency_weight,
        },
        "scoring_method": scoring_method,
        "minor_count": faithfulness["minor_count"] + fluency["minor_count"],
        "major_count": faithfulness["major_count"] + fluency["major_count"],
        "critical_count": critical_count,
        "text_score_5": text_score_5,
        "text_score_100_before_cap": text_score_100,
        "text_score_100": text_score_100,
    }
