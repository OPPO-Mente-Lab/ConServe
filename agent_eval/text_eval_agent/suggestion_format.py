# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

"""Normalize text-side error suggestions to '<left>'->'<right>'."""

from __future__ import annotations

import re

DELETE_MARKER = "[删除]"
_ARROW_SEP = "'->'"

_CHANGE_TO_RE = re.compile(
    r"(?is)change\s+(?:(?P<q1>['\"])(?P<from1>.*?)(?P=q1)\s+)?to\s+(?P<q2>['\"])(?P<to>.*?)(?P=q2)"
)
_CHANGE_TO_UNQUOTED_RE = re.compile(
    r"(?is)change\s+to\s+(?P<to>.+?)\s*[.。]?$"
)
_ZH_MODIFY_RE = re.compile(
    r"(?s)(?:将)?[“\"'](?P<from>.*?)[”\"']\s*(?:修改为|改为|改成)\s*[“\"'](?P<to>.*?)[”\"']"
)
_ZH_TO_ONLY_RE = re.compile(
    r"(?s)(?:修改为|改为|改成)\s*[“\"'](?P<to>.*?)[”\"']"
)
_REMOVE_RE = re.compile(
    r"(?is)^(?:remove|delete)\s+(?P<q>['\"])(?P<span>.*?)(?P=q)"
)
_ZH_DELETE_RE = re.compile(
    r"(?s)^删除\s*[“\"'](?P<span>.*?)[”\"']"
)
_QUOTED_RE = re.compile(r"[“\"'](.+?)[”\"']")


def format_suggestion(left: str, right: str) -> str:
    return f"'{left}'->'{right}'"


def resolve_suggestion_left(
    source_fragment: str | None,
    target_fragment: str | None,
) -> str:
    if source_fragment is not None and str(source_fragment).strip() != "":
        return str(source_fragment)
    if target_fragment is not None and str(target_fragment).strip() != "":
        return str(target_fragment)
    return ""


def parse_arrow_suggestion(suggestion: str) -> tuple[str, str] | None:
    text = suggestion.strip()
    if not (text.startswith("'") and text.endswith("'") and _ARROW_SEP in text):
        return None
    body = text[1:-1]
    left, right = body.split(_ARROW_SEP, 1)
    return left, right


def _is_deletion_right(right: str) -> bool:
    value = right.strip()
    return value in {"", DELETE_MARKER, "[delete]", "[Delete]", "[DELETE]"}


def _looks_like_pure_deletion(suggestion: str) -> bool:
    text = suggestion.strip()
    lower = text.lower()
    if _is_deletion_right(text):
        return True
    if lower.startswith("remove ") or lower.startswith("delete "):
        return True
    if text.startswith("删除"):
        # An instruction such as "delete A, keep only B" is a rewrite to B, not a pure deletion marker.
        if "保留" in text and len(_QUOTED_RE.findall(text)) >= 2:
            return False
        if "修改为" in text or "改为" in text or "改成" in text:
            return False
        return True
    return False


def _extract_right(suggestion: str) -> str | None:
    text = suggestion.strip().rstrip("。.")
    match = _CHANGE_TO_RE.search(text)
    if match:
        return match.group("to")
    match = _ZH_MODIFY_RE.search(text)
    if match:
        return match.group("to")
    match = _ZH_TO_ONLY_RE.search(text)
    if match:
        return match.group("to")
    match = _CHANGE_TO_UNQUOTED_RE.search(text)
    if match:
        return match.group("to").strip().strip("'\"“”")
    if text.startswith("删除") and "保留" in text:
        quotes = _QUOTED_RE.findall(text)
        if len(quotes) >= 2:
            return quotes[-1]
    if _looks_like_pure_deletion(text):
        return DELETE_MARKER
    if not re.match(r"(?is)^(change|remove|delete|将|删除|把)", text):
        return text
    quotes = _QUOTED_RE.findall(text)
    if quotes:
        return quotes[-1]
    return text


def normalize_suggestion(
    suggestion: str | None,
    *,
    source_fragment: str | None = None,
    target_fragment: str | None = None,
) -> str | None:
    if suggestion is None:
        return None
    raw = str(suggestion).strip()
    if not raw:
        return None

    left = resolve_suggestion_left(source_fragment, target_fragment)
    parsed = parse_arrow_suggestion(raw)
    if parsed is not None:
        parsed_left, parsed_right = parsed
        if not left:
            left = parsed_left
        right = DELETE_MARKER if _is_deletion_right(parsed_right) else parsed_right
        return format_suggestion(left, right)

    if _looks_like_pure_deletion(raw):
        match = _REMOVE_RE.search(raw) or _ZH_DELETE_RE.search(raw)
        if match and not left:
            left = match.group("span")
        elif not left and target_fragment is not None:
            left = str(target_fragment)
        return format_suggestion(left, DELETE_MARKER)

    right = _extract_right(raw)
    if right is None:
        return None
    if _is_deletion_right(right):
        right = DELETE_MARKER
    if not left and target_fragment is not None:
        left = str(target_fragment)
    return format_suggestion(left, right)
