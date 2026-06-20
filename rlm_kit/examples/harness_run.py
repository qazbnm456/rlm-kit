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


class Finding(BaseModel):
    title: str = Field(..., description="Short finding title.")
    severity: str = Field(..., description="low|medium|high|critical")


def _non_empty(text: str):
    return None if text.strip() else "empty response"


class SecurityTriage(RLMTask):
    signature = "evidence: str -> finding: Finding"
    output_field = "finding"
    output_model = Finding
    instructions = (
        "You are a security analyst. Use list_skills/read_skill to consult "
        "playbooks, then emit a Finding JSON."
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

    task = SecurityTriage(
        skills_dir=os.getenv("SKILLS_DIR", "./skills"),
        sub_lm=intercepted_sub,
    )

    trace_path = "./traces/run.jsonl"
    with TraceRecorder(trace_path, run_id="triage-001", meta={"task": "triage"}):
        finding = await task.arun(evidence="...redacted evidence blob...")
    print(finding.model_dump_json(indent=2))

    # The same trace doubles as an Agentic-RL dataset source.
    runs = group_by_run(load_events(trace_path))
    rl_records = export_rl(runs, reward=lambda events: 1.0)
    print(f"exported {len(rl_records)} RL step records")


if __name__ == "__main__":
    asyncio.run(main())
