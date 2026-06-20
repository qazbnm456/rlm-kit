"""Phase A — expose a directory of Skills to the RLM as tools.

A "Skill" here follows the common convention of a folder containing a ``SKILL.md``
(with optional YAML-ish frontmatter for ``name``/``description``), or a flat
``<name>.md`` file. We surface two tools to the main LM so it can decide, inside
the REPL, which skill to read — keeping control flow in the LM's hands (the run
stays an RLM, and the decision lands in the trajectory).

Pure stdlib; no dspy import.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

from .trace import record_tool_call

_PREVIEW_CHARS = 700   # head of a read skill recorded for inspection (a replay UI shows what was read)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: str

    def read(self) -> str:
        with open(self.path, "r", encoding="utf-8") as fh:
            return fh.read()


def _parse_frontmatter(text: str) -> dict:
    """Extract simple ``key: value`` pairs from a leading ``---`` block."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    meta: dict = {}
    for line in text[3:end].splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip().lower()] = value.strip()
    return meta


def discover_skills(skill_dir: str) -> list[Skill]:
    """Find skills under ``skill_dir``: ``*/SKILL.md`` folders and ``*.md`` files."""
    skills: list[Skill] = []
    if not os.path.isdir(skill_dir):
        return skills

    for entry in sorted(os.listdir(skill_dir)):
        full = os.path.join(skill_dir, entry)
        skill_md = os.path.join(full, "SKILL.md")
        if os.path.isdir(full) and os.path.isfile(skill_md):
            meta = _parse_frontmatter(_safe_read(skill_md))
            skills.append(
                Skill(
                    name=meta.get("name", entry),
                    description=meta.get("description", ""),
                    path=skill_md,
                )
            )
        elif os.path.isfile(full) and entry.endswith(".md"):
            meta = _parse_frontmatter(_safe_read(full))
            skills.append(
                Skill(
                    name=meta.get("name", entry[:-3]),
                    description=meta.get("description", ""),
                    path=full,
                )
            )
    return skills


def _safe_read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read(4096)  # frontmatter only; enough for metadata
    except OSError:
        return ""


def load_skills_as_tools(skill_dir: str) -> list[Callable]:
    """Return two tools — ``list_skills`` and ``read_skill`` — over ``skill_dir``.

    Both record a ``tool_call`` event so skill access shows up in the trajectory
    and the RL dataset.
    """
    skills = {s.name: s for s in discover_skills(skill_dir)}

    def list_skills() -> str:
        """List the available skills and their one-line descriptions."""
        if not skills:
            result = "No skills available."
        else:
            result = "\n".join(
                f"- {s.name}: {s.description or '(no description)'}"
                for s in skills.values()
            )
        record_tool_call("list_skills", args={}, result=result)
        return result

    def read_skill(name: str) -> str:
        """Read the full content of a named skill. Use list_skills first to see names."""
        skill: Optional[Skill] = skills.get(name)
        result = skill.read() if skill else f"No such skill: {name!r}."
        # `preview` (a head of the content) is for inspection — a trace reader / replay UI can show
        # WHAT was read, not just how long it was. `result_len` keeps the full size.
        record_tool_call("read_skill", args={"name": name}, result_len=len(result),
                         preview=result[:_PREVIEW_CHARS])
        return result

    return [list_skills, read_skill]
