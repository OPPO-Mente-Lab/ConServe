# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.error
import urllib.request
import uuid
from collections import OrderedDict
from typing import Protocol

from .runtime import ModelRuntimeConfig


class ModelClient(Protocol):
    def complete_json(self, component: str, messages: list[dict], runtime: ModelRuntimeConfig) -> dict:
        ...


class ModelClientError(RuntimeError):
    pass


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat completions client using stdlib only."""

    def __init__(self) -> None:
        # One instance is shared by all stages of one image evaluation. Keep
        # provider-reported usage separate from the parsed model JSON so it
        # cannot alter any agent schema.
        self.usage_records: list[dict[str, int | str | None]] = []
        # Persist an audit trail in the per-task result. It deliberately
        # excludes request headers, signatures, API keys, and other secrets.
        self.trace_records: list[dict] = []

    def complete_json(self, component: str, messages: list[dict], runtime: ModelRuntimeConfig) -> dict:
        endpoint = _endpoint_for_runtime(runtime)
        payload = {
            "model": runtime.model_for(component),
            "messages": messages,
            "temperature": runtime.temperature,
            "response_format": {"type": "json_object"},
        }
        trace = {
            "component": component,
            "model": payload["model"],
            "messages": messages,
            "request": {
                "temperature": runtime.temperature,
                "response_format": payload["response_format"],
            },
        }
        body = json.dumps(payload, ensure_ascii=False)
        request = urllib.request.Request(
            endpoint,
            data=body.encode("utf-8"),
            headers=_headers_for_runtime(runtime, body),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=runtime.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            trace["error"] = f"Model request failed: {exc}"
            self.trace_records.append(trace)
            raise ModelClientError(f"Model request failed: {exc}") from exc

        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            trace["raw_response"] = raw_body
            trace["error"] = "Model response is not valid JSON."
            self.trace_records.append(trace)
            raise ModelClientError("Model response is not valid JSON.") from exc
        trace["raw_response"] = data
        if "code" in data and data.get("code") != 0:
            trace["error"] = f"Model gateway error {data.get('code')}: {data.get('msg')}"
            self.trace_records.append(trace)
            raise ModelClientError(f"Model gateway error {data.get('code')}: {data.get('msg')}")

        usage = _usage_record(component, data)
        self.usage_records.append(usage)
        trace["token_usage"] = usage

        try:
            content = _message_content_from_response(data)
        except (KeyError, IndexError, TypeError) as exc:
            trace["error"] = "Model response did not contain choices[0].message.content."
            self.trace_records.append(trace)
            raise ModelClientError("Model response did not contain choices[0].message.content.") from exc

        try:
            parsed = parse_json_object(content)
        except ModelClientError as exc:
            trace["response_content"] = content
            trace["error"] = str(exc)
            self.trace_records.append(trace)
            raise
        trace["response_content"] = content
        trace["parsed_response"] = parsed
        self.trace_records.append(trace)
        return parsed


def _endpoint_for_runtime(runtime: ModelRuntimeConfig) -> str:
    suffix = "/completions" if runtime.provider == "signed_api" else "/chat/completions"
    return runtime.base_url.rstrip("/") + suffix


def _headers_for_runtime(runtime: ModelRuntimeConfig, body: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if runtime.provider == "signed_api":
        ak = runtime.resolve_ak()
        sk = runtime.resolve_sk()
        if not ak or not sk:
            raise ModelClientError("Missing signed API credentials. Provide runtime.ak/sk or runtime.ak_env/sk_env.")
        headers.update(
            {
                "recordId": str(uuid.uuid4()),
                "Authorization": _signed_api_signature(None, body, ak, sk),
            }
        )
        return headers

    api_key = runtime.resolve_api_key()
    if not api_key:
        raise ModelClientError("Missing API key. Provide runtime.api_key or runtime.api_key_env.")
    headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _signed_api_signature(params: dict | None, body: str, app_id: str, secret_key: str) -> str:
    auth_string_prefix = f"bot-auth-v1/{app_id}/{int(time.time() * 1000)}/"
    pieces = [auth_string_prefix]
    if params:
        ordered_params = OrderedDict(sorted(params.items()))
        pieces.extend([f"{key}={value}&" for key, value in ordered_params.items()])
    sign_str = "".join(pieces) + body
    signature = hmac.new(secret_key.encode("utf-8"), sign_str.encode("utf-8"), hashlib.sha256).hexdigest()
    return auth_string_prefix + signature


def _message_content_from_response(data: dict) -> str | dict:
    if "data" in data:
        return data["data"]["choices"][0]["message"]["content"]
    return data["choices"][0]["message"]["content"]


def _usage_record(component: str, response: dict) -> dict[str, int | str | None]:
    """Normalize optional model API token usage without estimating it."""
    container = response.get("data", response)
    usage = container.get("usage") if isinstance(container, dict) else None
    if not isinstance(usage, dict):
        usage = response.get("usage")
    if not isinstance(usage, dict):
        usage = {}

    def integer(*keys: str) -> int | None:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                return int(value)
        return None

    return {
        "component": component,
        "input_tokens": integer("prompt_tokens", "input_tokens"),
        "output_tokens": integer("completion_tokens", "output_tokens"),
        "total_tokens": integer("total_tokens"),
        "cached_input_tokens": integer("cached_tokens", "cached_input_tokens"),
    }


def parse_json_object(content: str | dict) -> dict:
    if isinstance(content, dict):
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ModelClientError("Model content is not valid JSON.")
        return json.loads(content[start : end + 1])
