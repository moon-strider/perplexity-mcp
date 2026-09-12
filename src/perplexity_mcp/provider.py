"""Bounded Sonar HTTP client with safe, actionable errors.

Only explicit HTTP 429/502/503/504 responses are automatically retried. A
timeout, disconnect, cancellation, or other ambiguous transport failure never
replays a possibly billable request. The selected timeout covers all attempts
and backoff together.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import aiohttp

from perplexity_mcp.config import Settings

RETRY_STATUSES = frozenset({429, 502, 503, 504})


class ProviderError(Exception):
    """Safe provider failure: neither request nor response text is included."""

    def __init__(
        self, code: str, message: str, *, retryable: bool = False, status: int | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.status = status


def _status_error(status: int) -> ProviderError:
    if status in (400, 422):
        return ProviderError(
            "invalid_request", "Perplexity rejected the request parameters.", status=status
        )
    if status == 401:
        return ProviderError(
            "authentication_error",
            "Perplexity authentication failed. Check PERPLEXITY_API_KEY.",
            status=status,
        )
    if status == 403:
        return ProviderError(
            "permission_denied",
            "Perplexity denied access. Check the account and model permissions.",
            status=status,
        )
    if status == 404:
        return ProviderError(
            "not_found", "The Perplexity endpoint or model was not found.", status=status
        )
    if status == 429:
        return ProviderError(
            "rate_limited",
            "Perplexity rate limited the request. Try again later.",
            retryable=True,
            status=status,
        )
    if 300 <= status < 400:
        return ProviderError(
            "redirect_not_allowed",
            "Perplexity returned a redirect; redirects are disabled to protect the API key.",
            status=status,
        )
    if status >= 500:
        return ProviderError(
            "provider_unavailable",
            "Perplexity is temporarily unavailable. Try again later.",
            retryable=True,
            status=status,
        )
    return ProviderError(
        "provider_error", "Perplexity returned an unsuccessful HTTP response.", status=status
    )


def _retry_delay(value: str | None, attempt: int) -> float:
    """Return Retry-After or bounded exponential backoff, without early retries."""
    fallback = min(0.5 * 2 ** (attempt - 1), 8.0)
    if not value:
        return fallback
    try:
        seconds = float(value)
        if math.isfinite(seconds) and seconds >= 0:
            return seconds
    except (ValueError, OverflowError):
        pass
    try:
        moment = parsedate_to_datetime(value)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return max(0.0, (moment - datetime.now(UTC)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return fallback


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON number")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite JSON number")
    return number


def _redact(value: Any, secret: str, depth: int = 0) -> Any:
    # Stay comfortably below the MCP/Pydantic serializer's recursion limit.
    # Body byte limits alone do not constrain tiny but deeply nested metadata.
    if depth > 64:
        raise ValueError("Response nesting exceeded the supported limit")
    if isinstance(value, str):
        # JSON permits escaped lone surrogates which Python retains, but the
        # MCP UTF-8 writer cannot serialize. Reject them before returning any
        # text, including nested metadata and object keys.
        value.encode("utf-8")
        return value.replace(secret, "[REDACTED]")
    if isinstance(value, list):
        return [_redact(item, secret, depth + 1) for item in value]
    if isinstance(value, dict):
        return {
            _redact(key, secret, depth + 1): _redact(item, secret, depth + 1)
            for key, item in value.items()
        }
    return value


def _normalize(data: Any, model: str, request_id: str | None) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError
    message = choices[0].get("message")
    if (
        not isinstance(message, dict)
        or not isinstance(message.get("content"), str)
        or not message["content"].strip()
    ):
        raise ValueError
    citations = [] if data.get("citations") is None else data["citations"]
    results = [] if data.get("search_results") is None else data["search_results"]
    usage = {} if data.get("usage") is None else data["usage"]
    actual_model = data.get("model", model)
    finish_reason = choices[0].get("finish_reason")
    if not isinstance(citations, list) or not all(isinstance(item, str) for item in citations):
        raise ValueError
    if not isinstance(results, list) or not all(isinstance(item, dict) for item in results):
        raise ValueError
    if not isinstance(usage, dict) or not isinstance(actual_model, str) or not actual_model.strip():
        raise ValueError
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise ValueError
    # Copy the citation list verbatim: deduplicating would corrupt inline [n] references.
    result = {
        "answer": message["content"],
        "citations": citations,
        "search_results": results,
        "model": actual_model,
        "usage": usage,
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
    }
    identifier = request_id or data.get("id")
    if identifier is not None:
        if not isinstance(identifier, str):
            raise ValueError
        result["request_id"] = identifier
    return result


class PerplexityClient:
    """One reusable HTTP session for the lifetime of an MCP server."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> PerplexityClient:
        if self._session is not None and not self._session.closed:
            raise RuntimeError("PerplexityClient is already open")
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None),
            trust_env=False,
            auto_decompress=False,
            headers={"Accept": "application/json", "Accept-Encoding": "identity"},
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _read_json(self, response: aiohttp.ClientResponse) -> Any:
        # Identity encoding makes the byte bound apply before any decompression.
        # An upstream ignoring Accept-Encoding is rejected rather than risking a
        # compressed response that expands past the memory limit.
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise ProviderError(
                "invalid_response", "Perplexity returned an unsupported response encoding."
            )
        limit = self.settings.response_limit
        if response.content_length is not None and response.content_length > limit:
            raise ProviderError(
                "response_too_large", "Perplexity response exceeded the configured byte limit."
            )
        body = bytearray()
        async for chunk in response.content.iter_chunked(65536):
            if len(body) + len(chunk) > limit:
                raise ProviderError(
                    "response_too_large", "Perplexity response exceeded the configured byte limit."
                )
            body.extend(chunk)
        try:
            # Sonar is UTF-8 JSON; accepting arbitrary response-supplied codecs
            # makes malformed upstream responses harder to diagnose safely.
            return json.loads(
                body.decode("utf-8"), parse_constant=_reject_constant, parse_float=_finite_float
            )
        except (ValueError, UnicodeError, RecursionError):
            raise ProviderError("invalid_response", "Perplexity returned invalid JSON.") from None

    async def complete(
        self,
        query: str,
        *,
        research: bool = False,
        recency: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        session = self._session
        if session is None or session.closed:
            raise RuntimeError("Use PerplexityClient as an async context manager")
        model = self.settings.research_model if research else self.settings.model
        default_tokens = (
            self.settings.research_max_tokens if research else self.settings.search_max_tokens
        )
        timeout = self.settings.research_timeout if research else self.settings.search_timeout
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": query}],
            "max_tokens": default_tokens if max_tokens is None else max_tokens,
            "web_search_options": {"search_context_size": "high"},
            "stream": False,
        }
        if recency is not None:
            payload["search_recency_filter"] = recency
        started = time.monotonic()
        deadline = started + timeout
        attempt = 0
        try:
            async with asyncio.timeout(timeout):
                while True:
                    attempt += 1
                    async with session.post(
                        self.settings.api_url,
                        json=payload,
                        headers={"Authorization": f"Bearer {self.settings.api_key}"},
                        allow_redirects=False,
                    ) as response:
                        if not 200 <= response.status < 300:
                            error = _status_error(response.status)
                            if (
                                response.status not in RETRY_STATUSES
                                or attempt > self.settings.max_retries
                            ):
                                raise error
                            delay = _retry_delay(response.headers.get("Retry-After"), attempt)
                            if delay >= deadline - time.monotonic():
                                raise error
                            # Release/close before waiting so retries do not hold a
                            # socket or consume an unbounded upstream error body.
                        else:
                            data = await self._read_json(response)
                            try:
                                result = _normalize(
                                    data, model, response.headers.get("x-request-id")
                                )
                                result = _redact(result, self.settings.api_key)
                            except (ValueError, TypeError, RecursionError):
                                raise ProviderError(
                                    "invalid_response",
                                    "Perplexity returned an unexpected response structure.",
                                ) from None
                            result["latency_seconds"] = round(time.monotonic() - started, 6)
                            result["attempts"] = attempt
                            return result
                    await asyncio.sleep(delay)
        except TimeoutError:
            raise ProviderError(
                "timeout",
                "The Perplexity request exceeded its total timeout. "
                "It was not retried after timing out.",
            ) from None
        except aiohttp.ClientError:
            raise ProviderError(
                "network_error",
                "The Perplexity connection failed. "
                "The request was not retried after the connection failure.",
            ) from None
