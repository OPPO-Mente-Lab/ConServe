# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import mimetypes
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Protocol

from .runtime import VisionRuntimeConfig


SUPPORTED_MIME_TYPES = {"image/png", "image/jpeg", "image/webp", "image/bmp"}

# A configured gateway may reject proxied inline-image requests before
# receiving the AK/SK signature. Use a per-client direct opener without
# changing global proxy environment variables.
_DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class VisionModelClient(Protocol):
    def complete_json(
        self,
        component: str,
        messages: list[dict],
        image_paths: list[Path],
        runtime: VisionRuntimeConfig,
    ) -> dict:
        ...


class VisionModelClientError(RuntimeError):
    pass


def make_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0]
    if mime_type not in SUPPORTED_MIME_TYPES:
        raise ValueError(f"Unsupported image format: {path.suffix or path.name}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_payload(
    component: str,
    messages: list[dict],
    image_data_urls: list[str],
    model: str,
    temperature: float,
) -> dict:
    del component
    if len(image_data_urls) not in {1, 2}:
        raise ValueError("Vision request requires exactly one or two images")
    copied_messages = [dict(message) for message in messages]
    user_indices = [index for index, message in enumerate(copied_messages) if message.get("role") == "user"]
    if not user_indices:
        raise ValueError("Vision request requires a user message")
    copied_messages[user_indices[-1]]["images"] = [
        {"url": image_data_url} for image_data_url in image_data_urls
    ]
    return {
        "model": model,
        "messages": copied_messages,
        "temperature": temperature,
    }


def serialize_payload(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def sign(
    body: str,
    app_id: str,
    secret_key: str,
    timestamp_ms: int | None = None,
) -> str:
    timestamp = int(time.time() * 1000) if timestamp_ms is None else timestamp_ms
    prefix = f"bot-auth-v1/{app_id}/{timestamp}/"
    signature = hmac.new(
        secret_key.encode("utf-8"),
        (prefix + body).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return prefix + signature


def extract_chat_content(response: dict) -> str | dict:
    container = response.get("data", response)
    try:
        content = container["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise VisionModelClientError("Gateway response does not contain message content") from error
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if isinstance(content, dict):
        return content
    if not isinstance(content, str) or not content.strip():
        raise VisionModelClientError("Chat message content must be non-empty")
    return content


def parse_json_object(content: str | dict) -> dict:
    if isinstance(content, dict):
        return content
    value = content.strip()
    if value.startswith("```"):
        first_newline = value.find("\n")
        if first_newline < 0 or not value.endswith("```"):
            raise VisionModelClientError("Incomplete JSON code fence")
        value = value[first_newline + 1 : -3].strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start < 0 or end <= start:
            raise VisionModelClientError("Model content is not valid JSON")
        try:
            parsed = json.loads(value[start : end + 1])
        except json.JSONDecodeError as error:
            raise VisionModelClientError("Model content is not valid JSON") from error
    if not isinstance(parsed, dict):
        raise VisionModelClientError("Model JSON must be an object")
    return parsed


class SignedVisionClient:
    def complete_json(
        self,
        component: str,
        messages: list[dict],
        image_paths: list[Path],
        runtime: VisionRuntimeConfig,
    ) -> dict:
        if runtime.provider != "signed_api":
            raise VisionModelClientError(
                f"Unsupported visual model provider: {runtime.provider}"
            )
        ak = runtime.resolve_ak()
        sk = runtime.resolve_sk()
        if not ak or not sk:
            raise VisionModelClientError(
                "Missing signed API credentials. Provide runtime.ak/sk or runtime.ak_env/sk_env."
            )
        paths = [Path(path) for path in image_paths]
        if len(paths) not in {1, 2} or any(not path.is_file() for path in paths):
            raise ValueError("Vision request requires exactly one or two existing images")
        payload = build_payload(
            component=component,
            messages=messages,
            image_data_urls=[make_data_url(path) for path in paths],
            model=runtime.model_for(component),
            temperature=runtime.temperature,
        )
        body = serialize_payload(payload)
        request = urllib.request.Request(
            _endpoint(runtime.base_url),
            data=body.encode("utf-8"),
            method="POST",
            headers={
                "Authorization": sign(body, ak, sk),
                "Content-Type": "application/json",
                "recordId": str(uuid.uuid4()),
            },
        )
        try:
            with _DIRECT_OPENER.open(request, timeout=runtime.timeout_seconds) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise VisionModelClientError(f"Vision model HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise VisionModelClientError(f"Vision model request failed: {error}") from error
        try:
            response_data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VisionModelClientError("Vision gateway response is not valid JSON") from error
        if "code" in response_data and response_data.get("code") != 0:
            raise VisionModelClientError(
                f"Vision gateway error {response_data.get('code')}: {response_data.get('msg')}"
            )
        return parse_json_object(extract_chat_content(response_data))


def _endpoint(base_url: str) -> str:
    value = base_url.rstrip("/")
    return value if value.endswith("/completions") else value + "/completions"
