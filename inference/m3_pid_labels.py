#!/usr/bin/env python3
# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

"""Shared multi-box renderer with non-occluding ``#pid`` labels."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


TARGET_BOX_COLOR_BGR = (0, 165, 255)  # orange
TARGET_BOX_THICK = 4
INDEX_LABEL_COLOR_BGR = (0, 0, 255)  # red
INDEX_LABEL_BG_BGR = (255, 255, 255)
_LABEL_HUG_GAP = TARGET_BOX_THICK + 1


def infer_paragraph_orientation(group: dict[str, Any] | None) -> str:
    """Infer horizontal/vertical text from OCR geometry."""
    if not group:
        return "horizontal"

    angles = group.get("angles") or []
    if angles:
        normalized = sorted(abs(int(angle)) % 180 for angle in angles)
        median = normalized[len(normalized) // 2]
        if 60 <= median <= 120:
            return "vertical"

    boxes = group.get("item_bboxes") or []
    if boxes:
        tall = sum(
            1
            for box in boxes
            if max(1, box[3] - box[1]) > max(1, box[2] - box[0]) * 1.3
        )
        if tall >= max(1, int(len(boxes) * 0.6)):
            return "vertical"
    return "horizontal"


def _rects_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    margin: int = 0,
) -> bool:
    return not (
        a[2] + margin <= b[0]
        or a[0] >= b[2] + margin
        or a[3] + margin <= b[1]
        or a[1] >= b[3] + margin
    )


def _overlap_area(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )


def _intersects_box_inner(
    label: tuple[int, int, int, int],
    box: tuple[int, int, int, int],
) -> bool:
    inner = (box[0] + 1, box[1] + 1, box[2] - 1, box[3] - 1)
    if inner[2] <= inner[0] or inner[3] <= inner[1]:
        return False
    return _rects_overlap(label, inner)


def _label_metrics(text: str, img_h: int, img_w: int) -> dict[str, Any]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, min(1.0, min(img_h, img_w) / 1200.0))
    thickness = max(2, int(round(scale * 2.5)))
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 3
    return {
        "font": font,
        "scale": scale,
        "thickness": thickness,
        "text_h": text_h,
        "pad": pad,
        "label_w": text_w + pad * 2,
        "label_h": text_h + baseline + pad * 2,
    }


def _batch_layout_axis(boxes: list[tuple[int, int, int, int]]) -> str:
    if len(boxes) < 2:
        return "vertical"
    centers_x = [(box[0] + box[2]) / 2 for box in boxes]
    centers_y = [(box[1] + box[3]) / 2 for box in boxes]
    spread_x = max(centers_x) - min(centers_x)
    spread_y = max(centers_y) - min(centers_y)
    return "horizontal" if spread_x > spread_y * 1.15 else "vertical"


def _side_priority(
    box: tuple[int, int, int, int],
    orientation: str,
    axis: str,
    img_w: int,
    img_h: int,
) -> list[str]:
    x1, y1, x2, y2 = box
    left_right = ["left", "right"] if x1 >= img_w - x2 else ["right", "left"]
    top_bottom = ["top", "bottom"] if y1 >= img_h - y2 else ["bottom", "top"]
    prefer_top = orientation == "vertical" or axis == "horizontal"
    return top_bottom + left_right if prefer_top else left_right + top_bottom


def _candidate_positions(
    side: str,
    box: tuple[int, int, int, int],
    metrics: dict[str, Any],
) -> list[tuple[int, int]]:
    x1, y1, x2, y2 = box
    label_w, label_h = metrics["label_w"], metrics["label_h"]
    fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
    if side in ("left", "right"):
        x = x1 - label_w - _LABEL_HUG_GAP if side == "left" else x2 + _LABEL_HUG_GAP
        span = max(0, y2 - y1 - label_h)
        return list(dict.fromkeys((x, int(round(y1 + span * f))) for f in fractions))

    y = y1 - label_h - _LABEL_HUG_GAP if side == "top" else y2 + _LABEL_HUG_GAP
    span = max(0, x2 - x1 - label_w)
    return list(dict.fromkeys((int(round(x1 + span * f)), y) for f in fractions))


def plan_pid_labels(
    pids: list[int],
    boxes: dict[int, tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
    orientation_by_pid: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Place labels inside the image, adjacent to their box, avoiding box interiors."""
    orientation_by_pid = orientation_by_pid or {}
    all_boxes = [boxes[pid] for pid in pids if pid in boxes]
    axis = _batch_layout_axis(all_boxes)
    placed: list[tuple[int, int, int, int]] = []
    plans: list[dict[str, Any]] = []

    for pid in pids:
        if pid not in boxes:
            continue
        own_box = boxes[pid]
        text = f"#{pid}"
        metrics = _label_metrics(text, img_h, img_w)
        sides = _side_priority(
            own_box,
            orientation_by_pid.get(pid, "horizontal"),
            axis,
            img_w,
            img_h,
        )
        candidates: list[tuple[int, tuple[int, int, int, int]]] = []
        for side_rank, side in enumerate(sides):
            for x, y in _candidate_positions(side, own_box, metrics):
                label = (x, y, x + metrics["label_w"], y + metrics["label_h"])
                if label[0] < 0 or label[1] < 0 or label[2] > img_w or label[3] > img_h:
                    continue
                box_overlap = sum(_overlap_area(label, box) for box in all_boxes if box != own_box)
                label_overlap = sum(_overlap_area(label, previous) for previous in placed)
                penalty = box_overlap + label_overlap + side_rank
                if box_overlap == 0 and label_overlap == 0 and not any(
                    _intersects_box_inner(label, box) for box in all_boxes
                ):
                    candidates = [(penalty, label)]
                    break
                candidates.append((penalty, label))
            if len(candidates) == 1 and candidates[0][0] == side_rank:
                break

        if candidates:
            _, label = min(candidates, key=lambda item: item[0])
        else:
            label_w, label_h = metrics["label_w"], metrics["label_h"]
            x = min(max(0, own_box[0]), max(0, img_w - label_w))
            y = min(
                max(0, own_box[1] - label_h - _LABEL_HUG_GAP),
                max(0, img_h - label_h),
            )
            label = (x, y, x + label_w, y + label_h)

        placed.append(label)
        plans.append({"pid": pid, "text": text, "rect": label, "metrics": metrics})
    return plans


def render_pid_framed_image(
    image_bgr: np.ndarray,
    pids: list[int],
    boxes: dict[int, tuple[int, int, int, int]],
    orientation_by_pid: dict[int, str] | None = None,
) -> np.ndarray:
    """Draw orange target boxes and red ``#pid`` labels outside box interiors."""
    canvas = image_bgr.copy()
    img_h, img_w = canvas.shape[:2]
    for pid in pids:
        if pid in boxes:
            x1, y1, x2, y2 = boxes[pid]
            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                TARGET_BOX_COLOR_BGR,
                TARGET_BOX_THICK,
            )

    for plan in plan_pid_labels(pids, boxes, img_w, img_h, orientation_by_pid):
        x1, y1, x2, y2 = plan["rect"]
        metrics = plan["metrics"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), INDEX_LABEL_BG_BGR, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), INDEX_LABEL_COLOR_BGR, 1)
        origin = (x1 + metrics["pad"], y1 + metrics["pad"] + metrics["text_h"])
        cv2.putText(
            canvas,
            plan["text"],
            origin,
            metrics["font"],
            metrics["scale"],
            INDEX_LABEL_COLOR_BGR,
            metrics["thickness"],
            cv2.LINE_AA,
        )
    return canvas
