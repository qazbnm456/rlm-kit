"""Schema-validation tool factory.

Produces a plain callable the RLM can invoke inside the REPL to check its draft
JSON against the expected pydantic schema before emitting a final answer — the
generalised form of the original app's ``validate_vulnerability_report``.
"""

from __future__ import annotations

from typing import Callable, Type

from pydantic import BaseModel


def make_schema_validator(model: Type[BaseModel]) -> Callable[[str], str]:
    """Return a tool that validates a JSON string against ``model``.

    The returned function's name and docstring are set so DSPy surfaces it to the
    model with useful tool metadata.
    """

    def validate(data_json_str: str) -> str:
        """Validate a JSON string against the expected output schema. Pass your
        generated JSON here before emitting it as the final answer."""
        try:
            model.model_validate_json(data_json_str)
            return "Validation successful. You may now output this JSON string."
        except Exception as exc:  # noqa: BLE001 — surfaced back to the model as text
            return f"Validation failed: {exc}"

    validate.__name__ = f"validate_{model.__name__.lower()}"
    validate.__qualname__ = validate.__name__
    return validate
