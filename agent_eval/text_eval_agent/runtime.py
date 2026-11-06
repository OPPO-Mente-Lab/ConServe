# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelRuntimeConfig:
    provider: str = "openai_compatible"
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    api_key_env: str | None = None
    ak: str | None = None
    sk: str | None = None
    ak_env: str | None = None
    sk_env: str | None = None
    default_model: str = "gpt-4.1-mini"
    agent_models: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 60
    temperature: float = 0.0

    @classmethod
    def from_dict(cls, data: dict | None) -> "ModelRuntimeConfig":
        data = data or {}
        data = dict(data)
        if "api_type" in data and "provider" not in data:
            data["provider"] = data["api_type"]
        if "api_base" in data and "base_url" not in data:
            data["base_url"] = data["api_base"]
        if "model" in data and "default_model" not in data:
            data["default_model"] = data["model"]
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in data.items() if key in allowed})

    def resolve_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None

    def resolve_ak(self) -> str | None:
        if self.ak:
            return self.ak
        if self.ak_env:
            return os.environ.get(self.ak_env)
        return None

    def resolve_sk(self) -> str | None:
        if self.sk:
            return self.sk
        if self.sk_env:
            return os.environ.get(self.sk_env)
        return None

    def model_for(self, component: str) -> str:
        return self.agent_models.get(component, self.default_model)

    def public_dict(self) -> dict:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "ak_env": self.ak_env,
            "sk_env": self.sk_env,
            "default_model": self.default_model,
            "agent_models": self.agent_models,
            "timeout_seconds": self.timeout_seconds,
            "temperature": self.temperature,
        }


