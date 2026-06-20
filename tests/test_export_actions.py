from rlm_kit.dataset import export_actions


def _run():
    return {
        "r1": [
            {"type": "run_start", "step_id": 0, "payload": {}},
            {"type": "main_step", "step_id": 1, "payload": {"reasoning": "plan", "code": "c"}},
            {"type": "tool_call", "step_id": 2, "payload": {
                "tool": "gen", "args": {"spec": "s"}, "reasoning": "qr",
                "raw": "id: x", "ok": True, "errors": []}},
            {"type": "sub_call", "step_id": 3, "payload": {
                "model": "gpt-5.5", "input": "ask", "processed": "answer"}},
            {"type": "main_step", "step_id": 4, "payload": {"reasoning": "assemble", "code": "c2"}},
        ]
    }


def test_export_actions_kinds_and_order():
    recs = export_actions(_run(), reward=lambda ev: 1.0)
    assert [r["kind"] for r in recs] == ["planner", "tool", "sub", "planner"]
    assert all(r["reward"] == 1.0 for r in recs)


def test_export_actions_tool_and_sub_payloads():
    recs = export_actions(_run())
    tool = recs[1]
    assert tool["tool"] == "gen" and tool["outcome"]["ok"] is True
    assert tool["action"]["input"] == {"spec": "s"}
    sub = recs[2]
    assert sub["action"]["input"] == "ask" and sub["outcome"]["output"] == "answer"


def test_export_actions_state_accumulates():
    recs = export_actions(_run())
    assert recs[0]["state"] == []
    assert len(recs[3]["state"]) == 3  # planner, tool, sub seen before the 2nd planner step


def test_export_actions_reward_none_when_unset():
    recs = export_actions(_run())
    assert all(r["reward"] is None for r in recs)
