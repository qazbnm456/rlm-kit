import importlib.util

import pytest

from rlm_kit.sandbox import (
    _JSON_LITERAL_ALIASES,
    SandboxSecurityError,
    build_interpreter,
)

# The pyodide/deno path now constructs dspy's PythonInterpreter (to inject the
# JSON-literal aliases), so it needs dspy at call time. Construction stays lazy —
# Deno is not spawned — so these run without a sandbox, just not without dspy.
_HAS_DSPY = importlib.util.find_spec("dspy") is not None
_needs_dspy = pytest.mark.skipif(not _HAS_DSPY, reason="dspy not installed")


@_needs_dspy
def test_pyodide_returns_alias_injecting_sandbox():
    interp = build_interpreter("pyodide")
    assert interp is not None
    # Same aliases, and constructing it did NOT spawn the Deno subprocess.
    assert interp._JSON_ALIASES == _JSON_LITERAL_ALIASES
    assert getattr(interp, "deno_process", None) is None
    interp.shutdown()


@_needs_dspy
def test_deno_returns_alias_injecting_sandbox():
    interp = build_interpreter("deno")
    assert interp._JSON_ALIASES == {"true": True, "false": False, "null": None}
    interp.shutdown()


@_needs_dspy
def test_none_defaults_to_alias_injecting_sandbox():
    interp = build_interpreter(None)
    assert interp is not None
    assert interp._JSON_ALIASES == _JSON_LITERAL_ALIASES
    interp.shutdown()


@_needs_dspy
def test_sandbox_execute_merges_json_aliases_into_variables(monkeypatch):
    """The override merges true/false/null into the variables dspy passes to the
    parent execute, so a JSON-trained model's `SUBMIT({"x": true})` resolves."""
    from dspy.primitives.python_interpreter import PythonInterpreter

    captured = {}

    def fake_super_execute(self, code, variables=None):
        captured["code"] = code
        captured["variables"] = variables
        return "ok"

    monkeypatch.setattr(PythonInterpreter, "execute", fake_super_execute)
    interp = build_interpreter("deno")
    assert interp.execute("SUBMIT({'x': true})", {"source": "s"}) == "ok"
    assert captured["variables"] == {
        "true": True,
        "false": False,
        "null": None,
        "source": "s",
    }


def test_local_without_optin_is_refused():
    with pytest.raises(SandboxSecurityError):
        build_interpreter("local")


def test_local_without_optin_refused_even_case_insensitive():
    with pytest.raises(SandboxSecurityError):
        build_interpreter("LOCAL")


def test_unknown_interpreter_raises_value_error():
    with pytest.raises(ValueError):
        build_interpreter("rce-please")


def test_mock_interpreter_has_execute():
    interp = build_interpreter("mock")
    assert interp is not None
    assert interp.execute("1+1") == ""
