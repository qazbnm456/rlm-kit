"""SSRF-aware fetch tooling for RLM tasks.

Tasks routinely need to pull remote content (web pages, docs, feeds, threat
intel). Handing an LLM-driven REPL an unrestricted fetcher is an SSRF liability:
the model can be steered — by the very untrusted content it is analysing — into
requesting internal services or cloud metadata endpoints.

``is_safe_url`` is a syntactic pre-flight guard (scheme + obvious
private/loopback/metadata targets). ``make_fetch_tool`` wraps a caller-supplied
(SYNC) fetcher with that guard, returning a sync tool ready to hand to
``RLMTask(tools=…)`` — dspy.RLM invokes tools synchronously, so the tool must be sync.

NOTE: a syntactic guard does not stop DNS rebinding (a public hostname that
resolves to a private address). Full protection requires re-checking the
*resolved* address at connection time inside your fetcher.
"""

from __future__ import annotations

import ipaddress
from typing import Callable
from urllib.parse import urlparse

from ..trace import record_tool_call

ALLOWED_SCHEMES = frozenset({"http", "https"})

# Hostnames that must never be fetched regardless of resolution.
_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "metadata.google.internal",
    }
)


def _hostname_is_blocked_literal_ip(hostname: str) -> bool:
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    # Block loopback, private, link-local (incl. 169.254.169.254 cloud metadata),
    # reserved, and unspecified ranges.
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_unspecified
        or ip.is_multicast
    )


def is_safe_url(url: str) -> bool:
    """Return True if ``url`` is an http(s) URL not pointing at an obvious
    internal/loopback/metadata target.

    Conservative by design: anything it cannot parse or recognise is rejected.
    """
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return False

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return False

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return False
    if hostname in _BLOCKED_HOSTNAMES:
        return False
    if hostname.endswith(".local") or hostname.endswith(".internal"):
        return False
    if _hostname_is_blocked_literal_ip(hostname):
        return False

    return True


# A fetcher maps a URL to its text content. SYNC: dspy.RLM calls tools synchronously
# (its sandbox bridge never awaits — an async tool serialises to a coroutine repr and
# never runs), so both the fetcher and the tool are sync. Wrap an async client yourself.
Fetcher = Callable[[str], str]


def make_fetch_tool(fetcher: Fetcher) -> Callable[[str], str]:
    """Wrap ``fetcher`` with the SSRF guard, returning a SYNC tool for dspy.RLM.

    SYNC because dspy.RLM's interpreter invokes tools synchronously (no await); an
    ``async def`` tool there returns an un-awaited coroutine the model never sees the
    result of, so ``fetcher`` must be sync too.

    The wrapper rejects unsafe URLs before the fetcher ever runs, and turns a fetcher
    error into a short string too (rather than raising), so the RLM can react to either
    as text. Each call records a ``tool_call`` carrying only the outcome (``ok`` +
    ``result_len`` / ``note``), NOT the fetched body — see below.
    """

    def fetch_url(url: str) -> str:
        """Fetch the text content at an http(s) URL. Internal, loopback, and
        cloud-metadata addresses are refused."""
        if not is_safe_url(url):
            record_tool_call(
                "fetch_url", args={"url": url}, ok=False,
                note="refused: not a permitted external http(s) URL",
            )
            return f"Refused: {url!r} is not a permitted external http(s) URL."
        try:
            result = fetcher(url)
        except Exception as exc:  # noqa: BLE001 — surface as text so the RLM can react
            record_tool_call(
                "fetch_url", args={"url": url}, ok=False,
                note=f"error: {type(exc).__name__}",
            )
            return f"Fetch error for {url!r}: {type(exc).__name__}: {str(exc)[:160]}"
        # Record status + size only, NOT the body. In an RLM the fetched text lands in
        # a REPL variable the model slices itself, so the body is bulk content that would
        # bloat the JSONL source-of-truth for no RL value (mirrors ``read_skill`` and
        # a consumer's fetch tool). Replay therefore serves this length, not the bytes.
        record_tool_call("fetch_url", args={"url": url}, ok=True, result_len=len(result))
        return result

    return fetch_url
