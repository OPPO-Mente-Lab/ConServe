# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

"""Standalone text-quality evaluation agent subsystem."""

from .orchestrator import evaluate_text_quality
from .image_pipeline import run_image_text_task
from .region_crop_pipeline import run_region_crop_text_task
from .text_pair_api import evaluate_text_pair, evaluate_text_pairs_batch

__all__ = [
    "evaluate_text_quality",
    "run_image_text_task",
    "run_region_crop_text_task",
    "evaluate_text_pair",
    "evaluate_text_pairs_batch",
]


