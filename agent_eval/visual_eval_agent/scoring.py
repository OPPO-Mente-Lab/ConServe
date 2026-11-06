# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from severity_scoring import score_severity_counts


DIMENSION_KEYS = {
    "visual.text_rendering": "text_rendering",
    "visual.layout": "layout",
    "visual.background": "background",
}

VISUAL_DIMENSION_WEIGHTS = {
    "text_rendering": 0.35,
    "layout": 0.35,
    "background": 0.30,
}


def score_visual_errors(errors: Iterable[dict[str, Any]]) -> dict[str, Any]:
    error_list = list(errors)
    global_counts = _severity_counts(error_list)
    dimension_scores: dict[str, dict[str, float | int]] = {}
    for dimension, key in DIMENSION_KEYS.items():
        dimension_errors = [
            error
            for error in error_list
            if error.get("primary_dimension", error.get("dimension")) == dimension
        ]
        counts = _severity_counts(dimension_errors)
        score_5 = score_severity_counts(
            counts["Minor"], counts["Major"], counts["Critical"]
        )
        dimension_scores[key] = {
            "minor_count": counts["Minor"],
            "major_count": counts["Major"],
            "critical_count": counts["Critical"],
            "score_5": score_5,
            "score_100": round(score_5 * 20, 4),
        }

    if global_counts["Critical"]:
        score_5 = 0.0
        scoring_method = "weighted_dimension_average_with_critical_cap"
    else:
        score_5 = round(
            sum(
                dimension_scores[key]["score_5"] * weight
                for key, weight in VISUAL_DIMENSION_WEIGHTS.items()
            ),
            4,
        )
        scoring_method = "weighted_dimension_average"
    return {
        "dimension_scores": dimension_scores,
        "dimension_weights": dict(VISUAL_DIMENSION_WEIGHTS),
        "scoring_method": scoring_method,
        "minor_count": global_counts["Minor"],
        "major_count": global_counts["Major"],
        "critical_count": global_counts["Critical"],
        "visual_score_5": score_5,
        "visual_score_100": round(score_5 * 20, 4),
    }


def _severity_counts(errors: Iterable[dict[str, Any]]) -> Counter:
    counts = Counter(str(error.get("severity")) for error in errors)
    return Counter(
        {
            "Minor": counts["Minor"],
            "Major": counts["Major"],
            "Critical": counts["Critical"],
        }
    )
