# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class VisionRuntimeConfig:
    provider: str = "signed_api"
    base_url: str = "https://api.example.invalid/v1"
    ak: str | None = None
    sk: str | None = None
    ak_env: str | None = "MODEL_SERVICE_AK"
    sk_env: str | None = "MODEL_SERVICE_SK"
    default_model: str = "gemini-3.1-pro-preview"
    agent_models: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 120.0
    temperature: float = 0.0
    optimize_image_transport: bool = True
    image_max_edge: int = 1600
    image_jpeg_quality: int = 88

    @classmethod
    def from_dict(cls, data: dict | None) -> "VisionRuntimeConfig":
        values = dict(data or {})
        if "api_type" in values and "provider" not in values:
            values["provider"] = values["api_type"]
        if "api_base" in values and "base_url" not in values:
            values["base_url"] = values["api_base"]
        if "model" in values and "default_model" not in values:
            values["default_model"] = values["model"]
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def resolve_ak(self) -> str | None:
        return self.ak or (os.environ.get(self.ak_env) if self.ak_env else None)

    def resolve_sk(self) -> str | None:
        return self.sk or (os.environ.get(self.sk_env) if self.sk_env else None)

    def model_for(self, component: str) -> str:
        return self.agent_models.get(component, self.default_model)

    def public_dict(self) -> dict:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "ak_env": self.ak_env,
            "sk_env": self.sk_env,
            "default_model": self.default_model,
            "agent_models": self.agent_models,
            "timeout_seconds": self.timeout_seconds,
            "temperature": self.temperature,
            "optimize_image_transport": self.optimize_image_transport,
            "image_max_edge": self.image_max_edge,
            "image_jpeg_quality": self.image_jpeg_quality,
        }
