"""Example: RLM-as-Harness — intercepted sub-LM + skills tools + traced run.

Wires every Phase A/B/C piece together (illustrative; needs real model creds and
a sandbox, so it is NOT imported by the test suite):

- a local/base model wrapped via intercept_sub_lm (validate + post-process),
- a Skills directory exposed to the main LM as tools (LM-decided),
- the whole run recorded to JSONL, then exported as an RL dataset.

Run as a script after exporting RLM_* env vars and pointing SKILLS_DIR at a
folder of skills.
"""

from __future__ import annotations

import asyncio
import os

import dspy
from pydantic import BaseModel, Field

from rlm_kit import (
    RLMConfig,
    RLMTask,
    TraceRecorder,
    configure,
    export_rl,
    group_by_run,
    intercept_sub_lm,
    load_events,
    load_skills_as_tools,
)


class Note(BaseModel):
    title: str = Field(..., description="Short note title.")
    takeaway: str = Field(..., description="the key takeaway, one line")


def _non_empty(text: str):
    return None if text.strip() else "empty response"


class Research(RLMTask):
    signature = "topic: str -> note: Note"
    output_field = "note"
    output_model = Note
    instructions = (
        "You are a research assistant. Use list_skills/read_skill to consult "
        "reference notes, then emit a Note JSON."
    )

    def __init__(self, skills_dir: str, **kw):
        # Skills become RLM tools; the LM decides when to read them.
        self.tools = load_skills_as_tools(skills_dir)
        super().__init__(**kw)


async def main() -> None:
    cfg = configure(RLMConfig.from_env())

    # Intercept the configured sub-model: trace every escalation + validate/post-process.
    base_sub = dspy.LM(cfg.sub_model, api_key=cfg.api_key, base_url=cfg.base_url)
    intercepted_sub = intercept_sub_lm(
        base_sub, validators=[_non_empty], postprocessors=[str.strip], name="local-sub"
    )

    task = Research(
        skills_dir=os.getenv("SKILLS_DIR", "./skills"),
        sub_lm=intercepted_sub,
    )

    trace_path = "./traces/run.jsonl"
    with TraceRecorder(trace_path, run_id="research-001", meta={"task": "research"}):
        note = await task.arun(topic="...the topic to research...")
    print(note.model_dump_json(indent=2))

    # The same trace doubles as an Agentic-RL dataset source.
    runs = group_by_run(load_events(trace_path))
    rl_records = export_rl(runs, reward=lambda events: 1.0)
    print(f"exported {len(rl_records)} RL step records")


if __name__ == "__main__":
    asyncio.run(main())
