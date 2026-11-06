# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


MINOR_DEDUCTION = 0.2
FIRST_MAJOR_SCORE = 3.0
ADDITIONAL_MAJOR_DEDUCTION = 0.5


def score_severity_counts(
    minor_count: int,
    major_count: int,
    critical_count: int,
) -> float:
    """Score one evaluation dimension with the shared 0-to-5 severity formula."""
    if min(minor_count, major_count, critical_count) < 0:
        raise ValueError("Severity counts must be non-negative")
    if critical_count:
        return 0.0
    if major_count:
        score = (
            FIRST_MAJOR_SCORE
            - ADDITIONAL_MAJOR_DEDUCTION * (major_count - 1)
            - MINOR_DEDUCTION * minor_count
        )
    else:
        score = 5.0 - MINOR_DEDUCTION * minor_count
    return round(max(0.0, score), 4)
