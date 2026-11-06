#!/usr/bin/env python3
# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

"""Flag inconsistent VLM source text using OCR overlap and confidence."""

from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Any


def normalize_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(char for char in normalized if char.isalnum())


def lcs_length(left: str, right: str) -> int:
    """Exact longest common subsequence length using O(min(n, m)) memory."""
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_char in left:
        current = [0]
        for index, right_char in enumerate(right, start=1):
            if left_char == right_char:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def compare_texts(ocr_text: Any, original_text: Any) -> dict[str, Any] | None:
    ocr = normalize_text(ocr_text)
    original = normalize_text(original_text)
    if not ocr or not original:
        return None
    ocr_len = len(ocr)
    original_len = len(original)
    common_char_count = len(set(ocr) & set(original))
    length_ratio = max(ocr_len, original_len) / min(ocr_len, original_len)
    orig_over_ocr = original_len / ocr_len
    lcs_len = 0 if common_char_count == 0 else lcs_length(ocr, original)
    lcs_similarity = 2.0 * lcs_len / (ocr_len + original_len)
    bag_overlap = sum((Counter(ocr) & Counter(original)).values())
    return {
        "ocr_normalized_len": ocr_len,
        "vlm_normalized_len": original_len,
        "length_ratio": length_ratio,
        "orig_over_ocr": orig_over_ocr,
        "common_char_count": common_char_count,
        "bag_overlap": bag_overlap,
        "cov_orig": bag_overlap / original_len,
        "cov_ocr": bag_overlap / ocr_len,
        "lcs_length": lcs_len,
        "lcs_similarity": lcs_similarity,
    }


def evaluate_paragraph(
    paragraph: dict[str, Any],
    lcs_threshold: float,
    length_ratio_threshold: float,
    ocr_confidence_threshold: float,
) -> tuple[bool, list[str], dict[str, Any] | None]:
    """Return whether to filter, reason codes and OCR/VLM overlap metrics."""
    if paragraph.get("should_translate") is not True or paragraph.get("vlm_ok") is False:
        return False, [], None
    metrics = compare_texts(paragraph.get("ocr_text"), paragraph.get("original_text"))
    if metrics is None:
        return False, [], None
    value = paragraph.get("ocr_mean_confidence")
    confidence = float(value) if isinstance(value, (int, float)) else None
    metrics["ocr_mean_confidence"] = confidence
    severe_length_anomaly = (
        metrics["lcs_similarity"] < lcs_threshold
        and metrics["length_ratio"] > length_ratio_threshold
    )
    if severe_length_anomaly:
        return True, ["low_lcs_and_abnormal_length"], metrics
    if metrics["common_char_count"] == 0 and confidence is not None and confidence > ocr_confidence_threshold:
        return True, ["no_common_character", "high_ocr_confidence"], metrics
    return False, [], metrics
