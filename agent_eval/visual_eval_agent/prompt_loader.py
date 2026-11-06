# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
PROMPT_DIR = PROJECT_DIR / "skills"
AGENT_SKILLS_DIR = PACKAGE_DIR / "agent_skills"


@lru_cache(maxsize=16)
def load_prompt(name: str) -> str:
    prompt_files = {
        "text_rendering": "text_rendering_agent.md",
        "layout": "layout_agent.md",
        "background": "background_agent.md",
        "visual_critic": "visual_critic_agent.md",
    }
    filename = prompt_files.get(name, f"{name}.md")
    path = PROMPT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8-sig").strip()


@lru_cache(maxsize=16)
def load_skill_instructions(skill_name: str, include_references: bool = False) -> str:
    skill_dir = AGENT_SKILLS_DIR / skill_name
    path = skill_dir / "SKILL.md"
    if not path.is_file():
        raise FileNotFoundError(f"Skill file not found: {path}")
    sections = [path.read_text(encoding="utf-8-sig").strip()]
    if include_references:
        reference_dir = skill_dir / "references"
        for reference in sorted(reference_dir.glob("*.md")):
            sections.append(
                f"\n# Loaded Reference: {reference.name}\n\n"
                + reference.read_text(encoding="utf-8-sig").strip()
            )
    return "\n\n".join(sections)
