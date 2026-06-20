import json
import threading
import types

from rlm_kit.trace import (
    EVENT_FINAL,
    EVENT_MAIN_STEP,
    EVENT_RESULT,
    EVENT_RUN_END,
    EVENT_RUN_START,
    EVENT_TOOL_CALL,
    TraceRecorder,
    current_recorder,
    group_by_run,
    load_events,
    record_tool_call,
)


def _counter():
    n = {"v": 0.0}

    def clock():
        n["v"] += 1.0
        return n["v"]

    return clock


def test_recorder_writes_jsonl_with_monotonic_steps(tmp_path):
    path = str(tmp_path / "trace.jsonl")
    with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:
        assert current_recorder() is rec
        rec.record("sub_call", {"x": 1})
        rec.record("tool_call", {"y": 2})
    assert current_recorder() is None  # reset on exit

    events = load_events(path)
    types_seen = [e["type"] for e in events]
    assert types_seen == [EVENT_RUN_START, "sub_call", "tool_call", EVENT_RUN_END]
    assert [e["step_id"] for e in events] == [0, 1, 2, 3]
    assert all(e["run_id"] == "r1" for e in events)
    assert all(e["schema"] == "rlm-kit/trace/v1" for e in events)


def test_on_event_observer_fires_live_for_every_event(tmp_path):
    # The live observer gets each event AS it is recorded (run_start, the calls, run_end) — what
    # a streaming UI uses to stream sandbox-invoked tool_calls that dspy's on_tool never sees.
    seen = []
    path = str(tmp_path / "trace.jsonl")
    with TraceRecorder(path, run_id="r1", clock=_counter(), on_event=seen.append) as rec:
        rec.record("tool_call", {"tool": "fetch_url"})
        rec.record("sub_call", {"x": 1})
    types = [e["type"] for e in seen]
    assert types == [EVENT_RUN_START, "tool_call", "sub_call", EVENT_RUN_END]   # live + in order
    assert seen == load_events(path)                                            # same events as the file


def test_on_event_observer_error_never_breaks_the_trace(tmp_path):
    path = str(tmp_path / "trace.jsonl")
    def boom(_):
        raise RuntimeError("observer blew up")
    with TraceRecorder(path, run_id="r1", clock=_counter(), on_event=boom) as rec:
        rec.record("tool_call", {"tool": "x"})                                  # must not raise
    assert [e["type"] for e in load_events(path)] == [EVENT_RUN_START, "tool_call", EVENT_RUN_END]


def test_run_end_records_error(tmp_path):
    path = str(tmp_path / "trace.jsonl")
    try:
        with TraceRecorder(path, run_id="r1", clock=_counter()):
            raise ValueError("boom")
    except ValueError:
        pass
    end = [e for e in load_events(path) if e["type"] == EVENT_RUN_END][0]
    assert end["payload"]["ok"] is False
    assert "boom" in end["payload"]["error"]


def test_record_main_trajectory_from_fake_prediction(tmp_path):
    path = str(tmp_path / "trace.jsonl")
    pred = types.SimpleNamespace(
        trajectory=[
            {"reasoning": "think", "code": "print(1)", "output": "1"},
            {"reasoning": "more", "code": "print(2)", "output": "2"},
        ],
        final_reasoning="done",
    )
    with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:
        rec.record_main_trajectory(pred)
        rec.record_result({"answer": 42})

    events = load_events(path)
    main = [e for e in events if e["type"] == EVENT_MAIN_STEP]
    assert len(main) == 2
    assert main[0]["payload"]["turn"] == 0
    assert main[1]["payload"]["code"] == "print(2)"
    final = [e for e in events if e["type"] == EVENT_FINAL][0]
    assert final["payload"]["final_reasoning"] == "done"
    result = [e for e in events if e["type"] == EVENT_RESULT][0]
    assert result["payload"]["output"] == {"answer": 42}


def test_main_step_ts_backfilled_from_live_capture(tmp_path):
    # The live per-turn stamps (captured while the run was in flight) override the post-hoc clock,
    # matched to the trajectory by reasoning — so a main_step's ts is WHEN it happened, not finalize.
    path = str(tmp_path / "trace.jsonl")
    pred = types.SimpleNamespace(
        trajectory=[
            {"reasoning": "think", "code": "c0", "output": "o0"},
            {"reasoning": "more", "code": "c1", "output": "o1"},
        ],
        final_reasoning="done",
    )
    with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:
        rec.begin_main_capture()
        rec.note_main_step("think", ts=100.5)   # live, as turn 0 was parsed
        rec.note_main_step("more", ts=120.0)    # live, as turn 1 was parsed
        rec.record_main_trajectory(pred)
    main = [e for e in load_events(path) if e["type"] == EVENT_MAIN_STEP]
    assert [e["payload"]["turn"] for e in main] == [0, 1]
    assert [e["ts"] for e in main] == [100.5, 120.0]    # live ts, NOT the _counter() fallback


def test_main_step_ts_falls_back_to_clock_without_capture(tmp_path):
    # No live capture (replay, or no callback wired) → ts is clock() exactly as before.
    path = str(tmp_path / "trace.jsonl")
    pred = types.SimpleNamespace(
        trajectory=[{"reasoning": "a", "code": "c", "output": "o"}], final_reasoning=None)
    with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:  # run_start consumes ts=1.0
        rec.record_main_trajectory(pred)
    main = [e for e in load_events(path) if e["type"] == EVENT_MAIN_STEP][0]
    assert main["ts"] == 2.0    # clock-driven fallback, unchanged behavior


def test_main_step_double_parse_resolves_to_first_stamp(tmp_path):
    # dspy fires the parse callback twice per turn (same reasoning) → consume the EARLIEST stamp.
    path = str(tmp_path / "trace.jsonl")
    pred = types.SimpleNamespace(
        trajectory=[{"reasoning": "r", "code": "c", "output": "o"}], final_reasoning=None)
    with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:
        rec.begin_main_capture()
        rec.note_main_step("r", ts=10.0)
        rec.note_main_step("r", ts=10.2)   # the duplicate fire
        rec.record_main_trajectory(pred)
    main = [e for e in load_events(path) if e["type"] == EVENT_MAIN_STEP][0]
    assert main["ts"] == 10.0


def test_begin_main_capture_resets_between_attempts(tmp_path):
    # A retry re-runs the RLM; only the final attempt is recorded, so a stamp from a prior attempt
    # must not leak into the recorded trajectory.
    path = str(tmp_path / "trace.jsonl")
    pred = types.SimpleNamespace(
        trajectory=[{"reasoning": "final-turn", "code": "c", "output": "o"}], final_reasoning=None)
    with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:
        rec.begin_main_capture()
        rec.note_main_step("stale-turn", ts=5.0)   # attempt 1
        rec.begin_main_capture()                    # attempt 2 starts → buffer cleared
        rec.note_main_step("final-turn", ts=200.0)
        rec.record_main_trajectory(pred)
    main = [e for e in load_events(path) if e["type"] == EVENT_MAIN_STEP][0]
    assert main["ts"] == 200.0   # the stale attempt-1 stamp did not match / leak


def test_record_main_trajectory_tolerates_missing_trajectory(tmp_path):
    path = str(tmp_path / "trace.jsonl")
    pred = types.SimpleNamespace()  # no trajectory, no final_reasoning
    with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:
        rec.record_main_trajectory(pred)  # must not raise
    events = load_events(path)
    assert any(e["type"] == EVENT_FINAL for e in events)
    assert not any(e["type"] == EVENT_MAIN_STEP for e in events)


def test_load_events_filters_by_run(tmp_path):
    path = str(tmp_path / "trace.jsonl")
    with TraceRecorder(path, run_id="a", clock=_counter()) as rec:
        rec.record("sub_call", {})
    with TraceRecorder(path, run_id="b", clock=_counter()) as rec:
        rec.record("sub_call", {})
    assert {e["run_id"] for e in load_events(path, run_id="a")} == {"a"}
    grouped = group_by_run(load_events(path))
    assert set(grouped) == {"a", "b"}


def test_result_serialises_pydantic(tmp_path):
    from pydantic import BaseModel

    class M(BaseModel):
        a: int

    path = str(tmp_path / "trace.jsonl")
    with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:
        rec.record_result(M(a=5))
    result = [e for e in load_events(path) if e["type"] == EVENT_RESULT][0]
    assert result["payload"]["output"] == {"a": 5}


def test_record_tool_call_emits_canonical_payload(tmp_path):
    path = str(tmp_path / "trace.jsonl")
    with TraceRecorder(path, run_id="r1", clock=_counter()):
        event = record_tool_call(
            "fetch_url", args={"url": "https://x"}, ok=True, result="body", note="ok"
        )
    # the helper returns the recorded event
    assert event["type"] == EVENT_TOOL_CALL
    tc = [e for e in load_events(path) if e["type"] == EVENT_TOOL_CALL][0]
    # the shape the replay/dataset readers consume: tool + args + merged extras
    assert tc["payload"] == {
        "tool": "fetch_url",
        "args": {"url": "https://x"},
        "ok": True,
        "result": "body",
        "note": "ok",
    }


def test_record_tool_call_omits_args_when_absent(tmp_path):
    path = str(tmp_path / "trace.jsonl")
    with TraceRecorder(path, run_id="r1", clock=_counter()):
        record_tool_call("validate", ok=False, errors=["bad"])
    tc = [e for e in load_events(path) if e["type"] == EVENT_TOOL_CALL][0]
    assert "args" not in tc["payload"]
    assert tc["payload"] == {"tool": "validate", "ok": False, "errors": ["bad"]}


def test_record_tool_call_noops_without_recorder():
    # No active recorder → no-op, returns None (a tool can call it unconditionally).
    assert current_recorder() is None
    assert record_tool_call("fetch_url", args={"url": "https://x"}, ok=True) is None


def test_record_is_thread_safe_under_concurrency(tmp_path):
    # llm_query_batched fans the wrapped sub_lm across threads → concurrent
    # sub_call records. The recorder must not race step_ids or interleave lines.
    path = str(tmp_path / "trace.jsonl")
    n = 200
    with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:
        threads = [
            threading.Thread(target=lambda i=i: rec.record("sub_call", {"i": i}))
            for i in range(n)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    events = load_events(path)
    sub = [e for e in events if e["type"] == "sub_call"]
    assert len(sub) == n  # no line lost or corrupted
    step_ids = [e["step_id"] for e in events]
    assert len(set(step_ids)) == len(step_ids)  # every step_id unique (no race)


def test_jsonl_is_valid_json_per_line(tmp_path):
    path = str(tmp_path / "trace.jsonl")
    with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:
        rec.record("sub_call", {"unicode": "漏洞"})
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            json.loads(line)  # raises if malformed


def test_recorder_scope_reestablishes_recorder_in_a_worker_thread(tmp_path):
    # A ThreadPoolExecutor worker does NOT inherit the recorder ContextVar (unlike an asyncio task), so
    # current_recorder() is None there — which is why dspy's llm_query_batched lost batched sub_calls.
    # recorder_scope re-establishes it so a record() from the worker lands in the trace.
    from concurrent.futures import ThreadPoolExecutor

    from rlm_kit.trace import current_recorder, recorder_scope

    path = str(tmp_path / "trace.jsonl")
    saw_none = {}
    with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:
        def work():
            saw_none["before"] = current_recorder() is None   # the bug: empty in a fresh worker thread
            with recorder_scope(rec):
                assert current_recorder() is rec
                rec.record("sub_call", {"i": 1})
        with ThreadPoolExecutor(max_workers=1) as ex:
            ex.submit(work).result()
    assert saw_none["before"] is True   # confirms the ContextVar did NOT propagate
    sub = [e for e in load_events(path) if e["type"] == "sub_call"]
    assert len(sub) == 1                # …and recorder_scope fixed it
