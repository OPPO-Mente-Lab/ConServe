# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import re
from typing import Any


_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_PREFIXES = {
    "evidence": "问题说明：",
    "suggestion": "修改建议：",
}


def localize_diagnostic_fields(value: Any) -> Any:
    """Ensure diagnostic prose has Chinese guidance without altering evidence fragments."""
    result = copy.deepcopy(value)
    _localize_in_place(result)
    return result


def _localize_in_place(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                key in _PREFIXES
                and isinstance(item, str)
                and item.strip()
                and not _CJK_PATTERN.search(item)
            ):
                value[key] = _PREFIXES[key] + item
            else:
                _localize_in_place(item)
    elif isinstance(value, list):
        for item in value:
            _localize_in_place(item)
