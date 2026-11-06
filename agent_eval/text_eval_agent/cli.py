# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .orchestrator import evaluate_text_quality
from .output_format import to_pretty_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate extracted text pairs for image translation quality.")
    parser.add_argument("input", help="Path to request JSON containing text_pairs.")
    parser.add_argument("--output", help="Optional path for result JSON.")
    args = parser.parse_args()

    request = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
    result = evaluate_text_quality(request)
    payload = to_pretty_json(result)

    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8-sig")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


