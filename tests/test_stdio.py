"""Blackbox tests: official MCP client -> subprocess -> a real loopback HTTP server."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fake_perplexity import OFFLINE_KEY, FakePerplexity, offline_environment
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import MCPError

ROOT = Path(__file__).resolve().parents[1]
SERVER_PYTHON = os.environ.get("PERPLEXITY_TEST_PYTHON", sys.executable)
TOOL_NAMES = {"perplexity_search_web", "perplexity_deep_research"}
PRIVATE_MARKER = "private-data-unique-marker"


@pytest.fixture
def provider():
    with FakePerplexity() as fake:
        yield fake


@asynccontextmanager
async def connected(provider, **overrides):
    env = offline_environment(provider.url, **overrides)
    parameters = StdioServerParameters(
        command=SERVER_PYTHON, args=["-m", "perplexity_mcp"], env=env, cwd=ROOT
    )
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr:
        async with stdio_client(parameters, errlog=stderr) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=10) as session:
                initialized = await session.initialize()
                yield session, initialized
        stderr.seek(0)
        logs = stderr.read()
        assert OFFLINE_KEY not in logs
        assert PRIVATE_MARKER not in logs
        assert "Traceback" not in logs
        assert "Unclosed client session" not in logs


def success(result):
    assert result.is_error is False
    data = result.structured_content
    assert isinstance(data, dict)
    assert isinstance(data["answer"], str) and data["answer"]
    assert data["citations"] == ["https://example.org/offline-source"]
    assert data["search_results"][0]["url"] == data["citations"][0]
    assert data["usage"]["total_tokens"] == 20
    assert isinstance(data["latency_seconds"], (float, int)) and data["latency_seconds"] >= 0
    assert isinstance(data["attempts"], int) and data["attempts"] >= 1
    assert any(
        content.type == "text" and data["answer"] in content.text for content in result.content
    )
    assert OFFLINE_KEY not in result.model_dump_json()
    return data


def failure(result):
    assert result.is_error is True
    assert result.content and all(item.type == "text" for item in result.content)
    assert isinstance(result.structured_content, dict)
    error = result.structured_content["error"]
    assert isinstance(error["code"], str) and error["code"]
    assert isinstance(error["message"], str) and error["message"]
    assert isinstance(error["retryable"], bool)
    assert OFFLINE_KEY not in result.model_dump_json()
    return error


async def test_discovery_prompts_and_both_tools_use_real_http(provider):
    async with connected(provider) as (session, initialized):
        assert initialized.capabilities.tools is not None
        assert initialized.capabilities.prompts is not None
        assert initialized.capabilities.resources is None
        listing = await session.list_tools()
        assert {tool.name for tool in listing.tools} == TOOL_NAMES
        for tool in listing.tools:
            assert tool.input_schema["additionalProperties"] is False
            assert tool.input_schema["required"] == ["query"]
            assert tool.input_schema["properties"]["recency"].get("default") is None
        prompts = await session.list_prompts()
        assert {prompt.name for prompt in prompts.prompts} == TOOL_NAMES
        for name in sorted(TOOL_NAMES):
            prompt = await session.get_prompt(name, {"query": "test question", "recency": "week"})
            text = " ".join(message.content.text for message in prompt.messages)
            assert "test question" in text and "week" in text
            result = success(await session.call_tool(name, {"query": f"Offline test of {name}"}))
            assert result["model"] == (
                "sonar" if name.endswith("search_web") else "sonar-deep-research"
            )
            assert result["finish_reason"] == "stop" and result["truncated"] is False
        await session.send_ping()

    assert len(provider.requests) == 2
    for request in provider.requests:
        assert request["path"] == "/v1/sonar"
        assert request["authorized"] is True
        assert request["payload"]["max_tokens"] == 8192
        assert type(request["payload"]["max_tokens"]) is int
        assert "search_recency_filter" not in request["payload"]
        assert "search_context_size" not in request["payload"]


async def test_explicit_recency_token_boundaries_unicode_and_truncation(provider):
    async with connected(provider) as (session, _):
        for recency, max_tokens in [
            ("hour", 16),
            ("day", 1),
            ("week", 8192),
            ("month", 32768),
            ("year", 65536),
        ]:
            success(
                await session.call_tool(
                    "perplexity_search_web",
                    {
                        "query": 'Привет 日本語 🧪\nquoted: "text"',
                        "recency": recency,
                        "max_tokens": max_tokens,
                    },
                )
            )
            payload = provider.requests[-1]["payload"]
            assert payload["search_recency_filter"] == recency
            assert payload["max_tokens"] == max_tokens
            assert payload["messages"][-1]["content"] == 'Привет 日本語 🧪\nquoted: "text"'
        data = success(
            await session.call_tool("perplexity_deep_research", {"query": "offline:truncated"})
        )
        assert data["finish_reason"] == "length" and data["truncated"] is True
        success(
            await session.call_tool(
                "perplexity_search_web",
                {"query": "Null is omitted", "recency": None, "max_tokens": None},
            )
        )
        assert "search_recency_filter" not in provider.requests[-1]["payload"]
        assert provider.requests[-1]["payload"]["max_tokens"] == 8192


async def test_invalid_arguments_and_unknown_tool_never_contact_provider(provider):
    invalid = [
        {},
        {"query": ""},
        {"query": " \n\t"},
        {"query": "x" * 32001},
        {"query": None},
        {"query": 4},
        {"query": True},
        {"query": []},
        {"query": "x", "unexpected": "ignored?"},
        {"query": "x", "recency": "test"},
        {"query": "x", "max_tokens": 0},
        {"query": "x", "max_tokens": 65537},
        {"query": "x", "max_tokens": "10"},
        {"query": "x", "max_tokens": True},
        {"query": "x", "max_tokens": 1.5},
    ]
    async with connected(provider) as (session, _):
        for name in sorted(TOOL_NAMES):
            for arguments in invalid:
                failure(await session.call_tool(name, arguments))
        failure(await session.call_tool("does_not_exist", {"query": "x"}))
        assert provider.requests == []
        success(await session.call_tool("perplexity_search_web", {"query": "x" * 32000}))
    assert len(provider.requests) == 1


async def test_invalid_prompts_do_not_call_provider_and_session_survives(provider):
    async with connected(provider) as (session, _):
        for name, args in [
            ("unknown_prompt", {"query": "x"}),
            ("perplexity_search_web", {}),
            ("perplexity_search_web", {"query": ""}),
            ("perplexity_deep_research", {"query": "x", "recency": "test"}),
            ("perplexity_search_web", {"query": "x", "extra": PRIVATE_MARKER}),
        ]:
            with pytest.raises(MCPError) as caught:
                await session.get_prompt(name, args)
            assert caught.value.code == -32602
            assert PRIVATE_MARKER not in caught.value.message
        await session.send_ping()
    assert provider.requests == []


async def test_provider_errors_are_safe_structured_results_and_do_not_break_session(provider):
    async with connected(provider) as (session, _):
        for status in [400, 401, 403, 404, 429, 500, 502, 503]:
            error = failure(
                await session.call_tool(
                    "perplexity_search_web", {"query": f"offline:status:{status}"}
                )
            )
            assert error["status"] == status
            assert error["retryable"] is (status == 429 or status >= 500)
        for query in [
            "offline:invalid-json",
            "offline:invalid-shape",
            "offline:invalid-unicode",
            "offline:invalid-metadata-unicode",
        ]:
            error = failure(await session.call_tool("perplexity_deep_research", {"query": query}))
            assert error["code"] == "invalid_response"
            await session.send_ping()
        success(await session.call_tool("perplexity_search_web", {"query": "Still operational"}))
    assert len(provider.requests) == 13


async def test_http_retry_is_observable_in_result_and_fixture(provider):
    async with connected(provider, PERPLEXITY_MAX_RETRIES="1") as (session, _):
        data = success(await session.call_tool("perplexity_search_web", {"query": "offline:retry"}))
        assert data["attempts"] == 2
    assert provider.counts["offline:retry"] == 2


async def test_concurrent_tools_keep_responses_associated_with_requests(provider):
    queries = [f"Concurrent query {index}" for index in range(8)]
    async with connected(provider) as (session, _):
        results = await asyncio.gather(
            *(
                session.call_tool(sorted(TOOL_NAMES)[index % 2], {"query": query})
                for index, query in enumerate(queries)
            )
        )
        for query, result in zip(queries, results, strict=True):
            assert query in success(result)["answer"]
    assert len(provider.requests) == len(queries)


async def test_http_timeout_returns_error_and_session_is_reusable(provider):
    async with connected(provider, PERPLEXITY_SEARCH_TIMEOUT="0.1") as (session, _):
        error = failure(
            await session.call_tool("perplexity_search_web", {"query": "offline:delay:0.5"})
        )
        assert error["code"] == "timeout"
        assert error["retryable"] is False
        success(await session.call_tool("perplexity_search_web", {"query": "After timeout"}))


async def test_cancel_inflight_tool_then_reuse_session(provider):
    async with connected(provider) as (session, _):
        pending = asyncio.create_task(
            session.call_tool("perplexity_search_web", {"query": "offline:delay:1"})
        )
        async with asyncio.timeout(5):
            while not provider.requests:
                await asyncio.sleep(0.01)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await session.send_ping()
        success(await session.call_tool("perplexity_search_web", {"query": "After cancellation"}))


@pytest.mark.parametrize("option", ["--help", "--version"])
def test_cli_metadata_works_without_key(option):
    env = {key: value for key, value in os.environ.items() if not key.startswith("PERPLEXITY_")}
    completed = subprocess.run(
        [SERVER_PYTHON, "-m", "perplexity_mcp", option],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip()
    assert "Traceback" not in completed.stderr


def test_missing_key_fails_cleanly_without_polluting_protocol_stdout():
    env = {key: value for key, value in os.environ.items() if not key.startswith("PERPLEXITY_")}
    completed = subprocess.run(
        [SERVER_PYTHON, "-m", "perplexity_mcp"],
        env=env,
        cwd=ROOT,
        input="",
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "PERPLEXITY_API_KEY" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_stdin_eof_exits_without_network_or_resource_leaks(provider):
    completed = subprocess.run(
        [SERVER_PYTHON, "-m", "perplexity_mcp"],
        env=offline_environment(provider.url),
        cwd=ROOT,
        input="",
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0
    assert completed.stdout == ""
    assert OFFLINE_KEY not in completed.stderr
    assert "Unclosed" not in completed.stderr
    assert provider.requests == []
