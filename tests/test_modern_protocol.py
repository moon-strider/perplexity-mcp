"""Raw 2026-07-28 wire regression, independent of the client's legacy handshake."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from fake_perplexity import OFFLINE_KEY, FakePerplexity, offline_environment

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_MARKER = "private-prompt-value-must-not-reach-logs"
PROTOCOL_KEY = "io.modelcontextprotocol/protocolVersion"
CAPABILITIES_KEY = "io.modelcontextprotocol/clientCapabilities"
SERVER_INFO_KEY = "io.modelcontextprotocol/serverInfo"
META = {
    PROTOCOL_KEY: "2026-07-28",
    CAPABILITIES_KEY: {},
    "io.modelcontextprotocol/clientInfo": {"name": "offline-protocol-audit", "version": "1"},
}


async def test_modern_envelopes_tools_prompts_and_errors_keep_stdio_usable():
    with FakePerplexity() as provider:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "perplexity_mcp",
            cwd=ROOT,
            env=offline_environment(provider.url),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        next_id = 0

        async def request(method, params=None, *, meta=META):
            nonlocal next_id
            next_id += 1
            payload = {
                "jsonrpc": "2.0",
                "id": next_id,
                "method": method,
                "params": {**(params or {}), "_meta": meta},
            }
            process.stdin.write(json.dumps(payload).encode() + b"\n")
            await process.stdin.drain()
            line = await asyncio.wait_for(process.stdout.readline(), timeout=5)
            assert line, "Server exited instead of responding"
            assert OFFLINE_KEY not in line.decode()
            assert PRIVATE_MARKER not in line.decode()
            response = json.loads(line)
            assert response["jsonrpc"] == "2.0" and response["id"] == next_id
            if "result" in response:
                assert response["result"]["resultType"] == "complete"
                assert response["result"]["_meta"][SERVER_INFO_KEY]["name"] == "perplexity-mcp"
            return response

        try:
            discovery = (await request("server/discover"))["result"]
            assert "2026-07-28" in discovery["supportedVersions"]
            assert set(discovery["capabilities"]) == {"tools", "prompts"}
            tools = (await request("tools/list"))["result"]["tools"]
            expected_names = {"perplexity_search_web", "perplexity_deep_research"}
            assert {tool["name"] for tool in tools} == expected_names
            for name in sorted(expected_names):
                result = (
                    await request("tools/call", {"name": name, "arguments": {"query": name}})
                )["result"]
                assert result["isError"] is False
                assert name in result["structuredContent"]["answer"]
                assert result["structuredContent"]["citations"]
            prompts = (await request("prompts/list"))["result"]["prompts"]
            assert {prompt["name"] for prompt in prompts} == expected_names
            valid_prompt = (
                await request(
                    "prompts/get",
                    {
                        "name": "perplexity_search_web",
                        "arguments": {"query": "a modern prompt", "recency": "hour"},
                    },
                )
            )["result"]
            assert "a modern prompt" in valid_prompt["messages"][0]["content"]["text"]

            # Expected user errors must be typed protocol errors, without traceback data.
            for name, arguments in [
                ("perplexity_search_web", {"query": "x", "extra": PRIVATE_MARKER}),
                ("unknown_prompt", {"query": "x"}),
            ]:
                failure = await request("prompts/get", {"name": name, "arguments": arguments})
                assert failure["error"]["code"] == -32602
            invalid_tool = (
                await request("tools/call", {"name": "perplexity_search_web", "arguments": {}})
            )["result"]
            assert invalid_tool["isError"] is True
            assert invalid_tool["structuredContent"]["error"]["code"] == "invalid_arguments"

            # Every modern request needs its envelope; a malformed request must not
            # break an already established stream or downgrade it to a legacy session.
            malformed = await request("tools/list", meta={PROTOCOL_KEY: "2026-07-28"})
            assert malformed["error"]["code"] == -32602
            assert (await request("tools/list"))["result"]["tools"] == tools

            # The SDK's JSON decoder rejects escaped lone surrogates before
            # request routing and discards that frame without an error envelope.
            # Assert the actual transport guarantee: the next valid frame works.
            process.stdin.write(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": "malformed-unicode-frame",
                        "method": "prompts/get",
                        "params": {
                            "name": "perplexity_search_web",
                            "arguments": {"query": "invalid " + chr(0xD800)},
                            "_meta": META,
                        },
                    }
                ).encode()
                + b"\n"
            )
            await process.stdin.drain()
            assert (await request("tools/list"))["result"]["tools"] == tools
        finally:
            process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                process.kill()
                await process.wait()
        logs = (await process.stderr.read()).decode()
        assert process.returncode == 0, logs
        assert OFFLINE_KEY not in logs
        assert PRIVATE_MARKER not in logs
        assert "Traceback" not in logs
        assert "Unclosed" not in logs
        assert len(provider.requests) == 2
        assert all(item["authorized"] for item in provider.requests)
