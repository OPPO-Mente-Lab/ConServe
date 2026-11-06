# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .policy import render_policy_prompt


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
AGENT_SKILLS_DIR = Path(__file__).resolve().parent / "agent_skills"


@lru_cache(maxsize=32)
def load_prompt(name: str, include_policy: bool = True) -> str:
    path = PROMPT_DIR / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    instructions = path.read_text(encoding="utf-8-sig").strip()
    return _with_policy(instructions) if include_policy else instructions


@lru_cache(maxsize=32)
def load_skill_instructions(skill_name: str, include_policy: bool = True) -> str:
    path = AGENT_SKILLS_DIR / skill_name / "SKILL.md"
    if not path.is_file():
        raise FileNotFoundError(f"Skill file not found: {path}")
    instructions = path.read_text(encoding="utf-8-sig").strip()
    return _with_policy(instructions) if include_policy else instructions


def _with_policy(instructions: str) -> str:
    return f"{instructions}\n\n{render_policy_prompt()}"

