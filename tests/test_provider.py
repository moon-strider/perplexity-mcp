import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import pytest
from aiohttp import web

from perplexity_mcp.config import Settings
from perplexity_mcp.provider import PerplexityClient, ProviderError, _retry_delay

KEY = "pplx-offline-secret"


def completion(**changes):
    data = {
        "id": "completion-123",
        "model": "sonar",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Answer [1] and [2]."},
                "finish_reason": "stop",
            }
        ],
        "citations": ["https://example.org/source", "https://example.org/source"],
        "search_results": [
            {"title": "A source", "url": "https://example.org/source", "date": "2026-09-01"}
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 7,
            "total_tokens": 17,
            "cost": {"total_cost": 0.001},
        },
    }
    return data | changes


@asynccontextmanager
async def serve(handler, **settings):
    application = web.Application()
    application.router.add_route("*", "/{path:.*}", handler)
    runner = web.AppRunner(application, shutdown_timeout=0.05)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    configured = Settings(api_key=KEY, api_url=f"http://127.0.0.1:{port}/v1/sonar", **settings)
    try:
        yield configured
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_search_payload_auth_sources_metadata_and_reused_session():
    requests = []
    transports = []

    async def handler(request):
        requests.append(
            {
                "json": await request.json(),
                "authorization": request.headers["Authorization"],
                "encoding": request.headers["Accept-Encoding"],
                "method": request.method,
                "path": request.path,
            }
        )
        transports.append(request.transport)
        return web.json_response(completion(), headers={"x-request-id": "request-456"})

    async with serve(handler, search_max_tokens=42) as settings:
        async with PerplexityClient(settings) as client:
            result = await client.complete("A real query")
            await client.complete("Second query", recency="week", max_tokens=123)
    assert requests[0] == {
        "json": {
            "model": "sonar",
            "messages": [{"role": "user", "content": "A real query"}],
            "max_tokens": 42,
            "web_search_options": {"search_context_size": "high"},
            "stream": False,
        },
        "authorization": f"Bearer {KEY}",
        "encoding": "identity",
        "method": "POST",
        "path": "/v1/sonar",
    }
    assert requests[1]["json"]["search_recency_filter"] == "week"
    assert requests[1]["json"]["max_tokens"] == 123
    assert transports[0] is transports[1]
    assert result["answer"] == "Answer [1] and [2]."
    assert result["citations"] == completion()["citations"]
    assert result["search_results"] == completion()["search_results"]
    assert result["usage"] == completion()["usage"]
    assert result["model"] == "sonar"
    assert result["request_id"] == "request-456"
    assert result["attempts"] == 1
    assert result["latency_seconds"] >= 0
    assert result["finish_reason"] == "stop"
    assert result["truncated"] is False


@pytest.mark.asyncio
async def test_research_uses_its_model_token_and_timeout_settings():
    received = []

    async def handler(request):
        received.append(await request.json())
        await asyncio.sleep(0.02)
        data = completion(model="sonar-deep-research")
        data["choices"][0]["finish_reason"] = "length"
        return web.json_response(data)

    async with serve(
        handler,
        model="sonar-pro",
        research_max_tokens=2048,
        search_timeout=0.001,
        research_timeout=1,
    ) as settings:
        async with PerplexityClient(settings) as client:
            result = await client.complete("Research this", research=True)
    assert received[0]["model"] == "sonar-deep-research"
    assert received[0]["max_tokens"] == 2048
    assert "search_recency_filter" not in received[0]
    assert result["truncated"] is True


@pytest.mark.asyncio
async def test_minimal_completion_is_supported():
    async def handler(request):
        return web.json_response({"choices": [{"message": {"content": "Answer"}}]})

    async with serve(handler, model="sonar-pro") as settings:
        async with PerplexityClient(settings) as client:
            result = await client.complete("query")
    assert result["model"] == "sonar-pro"
    assert result["usage"] == {}
    assert result["citations"] == []
    assert result["search_results"] == []
    assert result["finish_reason"] is None
    assert "request_id" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,code",
    [
        (400, "invalid_request"),
        (401, "authentication_error"),
        (403, "permission_denied"),
        (404, "not_found"),
        (422, "invalid_request"),
        (429, "rate_limited"),
        (500, "provider_unavailable"),
        (502, "provider_unavailable"),
        (503, "provider_unavailable"),
        (504, "provider_unavailable"),
        (418, "provider_error"),
    ],
)
async def test_http_errors_are_safe(status, code, caplog):
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return web.Response(status=status, text=KEY * 100, headers={"x-request-id": KEY})

    async with serve(handler, max_retries=0, response_limit=1024) as settings:
        async with PerplexityClient(settings) as client:
            with pytest.raises(ProviderError) as caught:
                await client.complete("private user query")
    assert caught.value.code == code
    assert caught.value.status == status
    assert calls == 1
    assert KEY not in str(caught.value)
    assert KEY not in repr(caught.value)
    assert KEY not in caplog.text
    assert "private user query" not in str(caught.value)


@pytest.mark.asyncio
async def test_redirect_does_not_forward_api_key_or_repeat_request():
    paths = []

    async def handler(request):
        paths.append(request.path)
        return web.Response(status=307, headers={"Location": "/stolen"})

    async with serve(handler) as settings:
        async with PerplexityClient(settings) as client:
            with pytest.raises(ProviderError) as caught:
                await client.complete("query")
    assert caught.value.code == "redirect_not_allowed"
    assert paths == ["/v1/sonar"]


@pytest.mark.asyncio
async def test_transient_responses_retry_then_succeed():
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls <= 2:
            return web.Response(status=(502 if calls == 1 else 429), headers={"Retry-After": "0"})
        return web.json_response(completion())

    async with serve(handler) as settings:
        async with PerplexityClient(settings) as client:
            result = await client.complete("query")
    assert calls == result["attempts"] == 3


@pytest.mark.asyncio
async def test_retry_count_is_bounded():
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return web.Response(status=503, headers={"Retry-After": "0"})

    async with serve(handler) as settings:
        async with PerplexityClient(settings) as client:
            with pytest.raises(ProviderError) as caught:
                await client.complete("query")
    assert calls == 3
    assert caught.value.code == "provider_unavailable"
    assert caught.value.retryable is True


@pytest.mark.asyncio
async def test_http_500_is_not_automatically_retried():
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return web.Response(status=500)

    async with serve(handler) as settings:
        async with PerplexityClient(settings) as client:
            with pytest.raises(ProviderError):
                await client.complete("query")
    assert calls == 1


@pytest.mark.asyncio
async def test_retry_after_larger_than_remaining_budget_does_not_retry_early():
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return web.Response(status=429, headers={"Retry-After": "3600"})

    async with serve(handler, search_timeout=0.1) as settings:
        async with PerplexityClient(settings) as client:
            started = time.monotonic()
            with pytest.raises(ProviderError) as caught:
                await client.complete("query")
            assert time.monotonic() - started < 0.5
    assert caught.value.code == "rate_limited"
    assert calls == 1


@pytest.mark.asyncio
async def test_timeout_budget_includes_retry_wait_and_next_attempt():
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return web.Response(status=503, headers={"Retry-After": "0.04"})
        await asyncio.sleep(0.2)
        return web.json_response(completion())

    async with serve(handler, search_timeout=0.1) as settings:
        async with PerplexityClient(settings) as client:
            started = time.monotonic()
            with pytest.raises(ProviderError) as caught:
                await client.complete("query")
            elapsed = time.monotonic() - started
    assert caught.value.code == "timeout"
    assert 0.075 <= elapsed < 0.5
    assert calls == 2


@pytest.mark.asyncio
async def test_slow_response_body_times_out_without_replay():
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        response = web.StreamResponse(headers={"Content-Type": "application/json"})
        await response.prepare(request)
        await response.write(b'{"choices":')
        await asyncio.sleep(0.2)
        return response

    async with serve(handler, search_timeout=0.03) as settings:
        async with PerplexityClient(settings) as client:
            with pytest.raises(ProviderError) as caught:
                await client.complete("query")
    assert caught.value.code == "timeout"
    assert calls == 1


@pytest.mark.asyncio
async def test_disconnect_does_not_replay_possibly_billable_request():
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        await request.read()
        request.transport.close()
        return web.Response()

    async with serve(handler) as settings:
        async with PerplexityClient(settings) as client:
            with pytest.raises(ProviderError) as caught:
                await client.complete("query")
    assert caught.value.code == "network_error"
    assert calls == 1


@pytest.mark.asyncio
async def test_cancellation_propagates_and_client_remains_usable():
    started = asyncio.Event()
    finish = asyncio.Event()
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await finish.wait()
        return web.json_response(completion())

    async with serve(handler) as settings:
        async with PerplexityClient(settings) as client:
            task = asyncio.create_task(client.complete("cancel me"))
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            finish.set()
            assert (await client.complete("next"))["answer"]
    assert calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        b"not JSON",
        b"<html>upstream error</html>",
        b"\xff",
        b'{"value": NaN}',
        b'{"value": Infinity}',
        b'{"value": 1e999}',
        b"[" * 2000 + b"]" * 2000,
    ],
)
async def test_invalid_json_is_safe_and_not_retried(body):
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return web.Response(body=body)

    async with serve(handler) as settings:
        async with PerplexityClient(settings) as client:
            with pytest.raises(ProviderError) as caught:
                await client.complete("query")
    assert caught.value.code == "invalid_response"
    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        {},
        {"choices": []},
        {"choices": [None]},
        {"choices": [{"message": None}]},
        {"choices": [{"message": {"content": " "}}]},
        completion(citations=[1]),
        completion(search_results=["bad"]),
        completion(usage=[]),
        completion(model=1),
        completion(id=1),
        {"choices": [{"message": {"content": "ok"}, "finish_reason": 1}]},
    ],
)
async def test_malformed_completion_is_rejected(data):
    async def handler(request):
        return web.json_response(data)

    async with serve(handler) as settings:
        async with PerplexityClient(settings) as client:
            with pytest.raises(ProviderError) as caught:
                await client.complete("query")
    assert caught.value.code == "invalid_response"


@pytest.mark.asyncio
async def test_documented_nullable_optional_metadata():
    async def handler(request):
        return web.json_response(completion(citations=None, search_results=None, usage=None))

    async with serve(handler) as settings:
        async with PerplexityClient(settings) as client:
            result = await client.complete("query", recency="hour")
    assert result["citations"] == []
    assert result["search_results"] == []
    assert result["usage"] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("chunked", [False, True])
async def test_response_byte_limit_with_and_without_content_length(chunked):
    async def handler(request):
        if not chunked:
            return web.Response(body=b"x" * 2048)
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b"x" * 1024)
        await response.write(b"x")
        await response.write_eof()
        return response

    async with serve(handler, response_limit=1024) as settings:
        async with PerplexityClient(settings) as client:
            with pytest.raises(ProviderError) as caught:
                await client.complete("query")
    assert caught.value.code == "response_too_large"


@pytest.mark.asyncio
async def test_compression_is_rejected_to_enforce_pre_decompression_limit():
    async def handler(request):
        return web.Response(body=b"fake gzip", headers={"Content-Encoding": "gzip"})

    async with serve(handler) as settings:
        async with PerplexityClient(settings) as client:
            with pytest.raises(ProviderError) as caught:
                await client.complete("query")
    assert caught.value.code == "invalid_response"


@pytest.mark.asyncio
async def test_success_response_cannot_echo_api_key_in_any_field(caplog):
    async def handler(request):
        data = completion(
            model=KEY,
            citations=[f"https://example.org/{KEY}"],
            search_results=[{"title": KEY, KEY: {"nested": [KEY]}}],
            usage={KEY: KEY},
        )
        data["choices"][0]["message"]["content"] = f"Secret: {KEY}"
        data["choices"][0]["finish_reason"] = KEY
        return web.json_response(data, headers={"x-request-id": KEY})

    async with serve(handler) as settings:
        async with PerplexityClient(settings) as client:
            result = await client.complete("query")
    assert KEY not in json.dumps(result)
    assert "[REDACTED]" in result["answer"]
    assert KEY not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field", ["answer", "citations", "search_results", "usage", "model", "finish_reason"]
)
async def test_lone_surrogate_in_any_retained_string_is_rejected(field):
    async def handler(request):
        data = completion()
        if field == "answer":
            data["choices"][0]["message"]["content"] = "bad \ud800"
        elif field == "finish_reason":
            data["choices"][0]["finish_reason"] = "\udfff"
        elif field == "citations":
            data["citations"] = ["https://example.org/\ud800"]
        elif field == "search_results":
            data["search_results"] = [{"nested": {"key": "\ud800"}}]
        elif field == "usage":
            data["usage"] = {"\ud800": "value"}
        else:
            data["model"] = "\ud800"
        return web.Response(body=json.dumps(data, ensure_ascii=True).encode("ascii"))

    async with serve(handler) as settings:
        async with PerplexityClient(settings) as client:
            with pytest.raises(ProviderError) as caught:
                await client.complete("query")
    assert caught.value.code == "invalid_response"


@pytest.mark.asyncio
async def test_valid_escaped_surrogate_pair_becomes_emoji():
    async def handler(request):
        return web.Response(body=rb'{"choices":[{"message":{"content":"Good \ud83d\ude00"}}]}')

    async with serve(handler) as settings:
        async with PerplexityClient(settings) as client:
            result = await client.complete("query")
    assert result["answer"] == "Good 😀"


@pytest.mark.asyncio
@pytest.mark.parametrize("depth,accepted", [(10, True), (300, False)])
async def test_retained_metadata_depth_stays_within_mcp_serialization_limit(depth, accepted):
    nested = "value"
    for _ in range(depth):
        nested = [nested]

    async def handler(request):
        return web.json_response(completion(usage={"other": nested}))

    async with serve(handler) as settings:
        async with PerplexityClient(settings) as client:
            if accepted:
                result = await client.complete("query")
                assert result["usage"]["other"] == nested
            else:
                with pytest.raises(ProviderError) as caught:
                    await client.complete("query")
                assert caught.value.code == "invalid_response"


@pytest.mark.asyncio
async def test_lifecycle_close_is_idempotent_and_reopening_works():
    async def handler(request):
        return web.json_response(completion())

    async with serve(handler) as settings:
        client = PerplexityClient(settings)
        with pytest.raises(RuntimeError):
            await client.complete("before opening")
        async with client:
            with pytest.raises(RuntimeError):
                await client.__aenter__()
            await client.complete("opened")
        await client.close()
        with pytest.raises(RuntimeError):
            await client.complete("after closing")
        async with client:
            assert (await client.complete("reopened"))["answer"]


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, 0.5),
        ("", 0.5),
        ("garbage", 0.5),
        ("-1", 0.5),
        ("nan", 0.5),
        ("inf", 0.5),
        ("0", 0),
        ("2.5", 2.5),
        ("999999", 999999),
    ],
)
def test_retry_after_numeric_and_invalid_values(value, expected):
    assert _retry_delay(value, 1) == expected


def test_retry_after_http_dates_and_bounded_fallback():
    future = format_datetime(datetime.now(UTC) + timedelta(seconds=30))
    assert 28 <= _retry_delay(future, 1) <= 30
    assert _retry_delay("Wed, 21 Oct 2015 07:28:00 GMT", 1) == 0
    assert _retry_delay(None, 2) == 1
    assert _retry_delay(None, 5) == 8
    assert _retry_delay(None, 10) == 8
