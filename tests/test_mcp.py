"""MCP client tests — the async->sync bridge is the load-bearing mechanic, so these run a REAL
stdio MCP server (a tiny FastMCP echo) as a subprocess and drive it through ``mcp_tools``. Skips
when the optional ``mcp`` extra (or dspy) is absent, like the other dspy-bearing tests."""

import sys

import pytest

pytest.importorskip("mcp")
pytest.importorskip("dspy")

from rlm_kit.mcp import _args_from_schema, _result_text, mcp_tools  # noqa: E402
from rlm_kit.trace import TraceRecorder, load_events  # noqa: E402

# A minimal stdio MCP server: one `echo` tool. Written to a temp file and spawned per test.
_ECHO_SERVER = '''
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo-test")


@mcp.tool()
def echo(text: str) -> str:
    """Echo the text back, prefixed."""
    return f"echo: {text}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
'''


def _server(tmp_path):
    p = tmp_path / "echo_server.py"
    p.write_text(_ECHO_SERVER)
    return {"command": sys.executable, "args": [str(p)]}


# ---- pure helpers --------------------------------------------------------

def test_args_from_schema_extracts_properties():
    assert _args_from_schema({"type": "object", "properties": {"text": {"type": "string"}}}) == {
        "text": {"type": "string"}
    }
    assert _args_from_schema(None) == {}
    assert _args_from_schema({"no": "properties"}) == {}


def test_result_text_joins_text_blocks_and_flags_errors():
    import types

    block = types.SimpleNamespace(text="hello")
    ok = types.SimpleNamespace(content=[block], structuredContent=None, isError=False)
    assert _result_text(ok) == "hello"
    err = types.SimpleNamespace(content=[block], structuredContent=None, isError=True)
    assert _result_text(err).startswith("[tool reported an error]")
    structured = types.SimpleNamespace(content=[], structuredContent={"n": 1}, isError=False)
    assert '"n": 1' in _result_text(structured)


# ---- the bridge: real stdio server, sync call, teardown ------------------

def test_mcp_tools_discovers_and_calls_sync(tmp_path):
    with mcp_tools(_server(tmp_path), timeout=30) as tools:
        assert [t.name for t in tools] == ["echo"]
        echo = tools[0]
        assert echo.args["text"]["type"] == "string"            # inputSchema mapped for the RLM
        # THE load-bearing assertion: a SYNC call drives the async session across the bridge.
        assert echo(text="hi") == "echo: hi"
        assert echo(text="again") == "echo: again"              # session stays alive across calls


def test_mcp_tool_call_is_traced(tmp_path):
    p = tmp_path / "t.jsonl"
    with TraceRecorder(str(p), run_id="r"):
        with mcp_tools(_server(tmp_path), timeout=30) as tools:
            tools[0](text="hi")
    calls = [e for e in load_events(str(p), "r")
             if e["type"] == "tool_call" and e["payload"]["tool"] == "echo"]
    assert calls and calls[0]["payload"]["ok"] is True
    assert "echo: hi" in calls[0]["payload"]["preview"]


def test_mcp_tools_teardown_leaves_no_live_thread(tmp_path):
    import threading

    before = {t.name for t in threading.enumerate()}
    with mcp_tools(_server(tmp_path), timeout=30) as tools:
        tools[0](text="x")
    # the background MCP thread is joined on exit — no leaked "rlm-kit-mcp" thread.
    after = {t.name for t in threading.enumerate()}
    assert "rlm-kit-mcp" not in (after - before)


def test_bad_server_spec_raises_and_cleans_up():
    import threading

    before = {t.name for t in threading.enumerate()}
    with pytest.raises(ValueError):
        with mcp_tools({"nonsense": 1}, timeout=5):
            pass
    # start() failed, but mcp_tools' finally still closed the bridge — no leaked background thread.
    assert "rlm-kit-mcp" not in ({t.name for t in threading.enumerate()} - before)
