# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any


def to_pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def jsonl_to_pretty_json(content: str) -> str:
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    return to_pretty_json(rows)
