"""Strict MCP tools backed by a bounded Perplexity HTTP client."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.lowlevel.server import ServerRequestContext
from mcp.server.models import InitializationOptions
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from . import __version__
from .config import Settings
from .provider import PerplexityClient, ProviderError

SEARCH = "perplexity_search_web"
RESEARCH = "perplexity_deep_research"
RECENCIES = ("hour", "day", "week", "month", "year")


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=32000)
    recency: Literal["hour", "day", "week", "month", "year"] | None = None
    max_tokens: int | None = Field(default=None, ge=1, le=65536)

    @field_validator("query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        value.encode("utf-8")
        if not value.strip():
            raise ValueError("query must contain non-whitespace text")
        return value


class PromptArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=32000)
    recency: Literal["hour", "day", "week", "month", "year"] | None = None

    @field_validator("query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        value.encode("utf-8")
        if not value.strip():
            raise ValueError("query must contain non-whitespace text")
        return value


def error_result(
    code: str, message: str, *, retryable: bool = False, status: int | None = None
) -> types.CallToolResult:
    error: dict[str, Any] = {"code": code, "message": message, "retryable": retryable}
    if status is not None:
        error["status"] = status
    return types.CallToolResult(
        isError=True,
        content=[types.TextContent(type="text", text=message)],
        structuredContent={"error": error},
    )


def format_result(result: dict[str, Any]) -> str:
    text = result["answer"]
    citations = result.get("citations", [])
    if citations:
        text += "\n\nSources:\n" + "\n".join(
            f"[{index}] {url}" for index, url in enumerate(citations, 1)
        )
    elif result.get("search_results"):
        # Unnumbered fallback: no invented correspondence with answer citations.
        text += "\n\nSearch results:\n" + "\n".join(
            f"- {item.get('title', '')}: {item['url']}"
            for item in result["search_results"]
            if item.get("url")
        )
    if result.get("truncated"):
        text += (
            "\n\nWarning: the provider stopped at the output token limit; "
            "this answer is incomplete."
        )
    metadata = {
        key: result.get(key)
        for key in ("model", "usage", "finish_reason", "latency_seconds", "attempts")
    }
    text += "\n\nRequest metadata:\n" + json.dumps(metadata, ensure_ascii=False, allow_nan=False)
    return text


def create_server(settings: Settings) -> Server:
    @asynccontextmanager
    async def lifespan(_: Server) -> AsyncIterator[PerplexityClient]:
        async with PerplexityClient(settings) as client:
            yield client

    async def list_tools(
        context: ServerRequestContext, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=name,
                    description=description,
                    inputSchema=ToolArguments.model_json_schema(),
                    annotations=types.ToolAnnotations(
                        readOnlyHint=True,
                        destructiveHint=False,
                        idempotentHint=False,
                        openWorldHint=True,
                    ),
                )
                for name, description in (
                    (
                        SEARCH,
                        "Search the web and answer with cited sources. Optional recency filter; "
                        "omitting it searches without a time restriction. "
                        "Uses the configured search model.",
                    ),
                    (
                        RESEARCH,
                        "Run a longer Perplexity research request with cited sources. "
                        "Can take several minutes and costs more than a normal search. "
                        "Optional recency filter; max_tokens limits the generated answer.",
                    ),
                )
            ]
        )

    async def call_tool(
        context: ServerRequestContext, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        name, arguments = params.name, params.arguments or {}
        if name not in (SEARCH, RESEARCH):
            return error_result(
                "unknown_tool", "Unknown tool. Use tools/list to discover available tools."
            )
        try:
            args = ToolArguments.model_validate(arguments)
        except ValidationError:
            return error_result(
                "invalid_arguments",
                "Invalid arguments: query must contain 1–32000 characters of nonblank text; "
                "recency must be hour, day, week, month, year or null; "
                "max_tokens must be an integer from 1 to 65536 or null. "
                "Extra arguments are rejected.",
            )
        try:
            client = context.lifespan_context
            result = await client.complete(
                args.query,
                research=name == RESEARCH,
                recency=args.recency,
                max_tokens=args.max_tokens,
            )
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=format_result(result))],
                structuredContent=result,
            )
        except ProviderError as exc:
            return error_result(exc.code, exc.message, retryable=exc.retryable, status=exc.status)
        except Exception:
            # Neither provider bodies nor user queries belong in server error logs.
            logging.error("Unexpected request failure; details withheld to protect request data")
            return error_result("internal_error", "An unexpected server error occurred.")

    async def list_prompts(
        context: ServerRequestContext, params: types.PaginatedRequestParams | None
    ) -> types.ListPromptsResult:
        return types.ListPromptsResult(
            prompts=[
                types.Prompt(
                    name=name,
                    description=description,
                    arguments=[
                        types.PromptArgument(
                            name="query", description="Question or topic", required=True
                        ),
                        types.PromptArgument(
                            name="recency", description="Optional: hour, day, week, month or year"
                        ),
                    ],
                )
                for name, description in (
                    (SEARCH, "Prepare a web search request"),
                    (RESEARCH, "Prepare a research request"),
                )
            ]
        )

    async def get_prompt(
        context: ServerRequestContext, params: types.GetPromptRequestParams
    ) -> types.GetPromptResult:
        name, arguments = params.name, params.arguments
        if name not in (SEARCH, RESEARCH):
            raise MCPError(
                types.INVALID_PARAMS,
                "Unknown prompt. Use prompts/list to discover available prompts.",
            )
        try:
            args = PromptArguments.model_validate(arguments or {})
        except ValidationError:
            raise MCPError(
                types.INVALID_PARAMS,
                "Invalid prompt arguments: provide a nonblank query and an optional "
                "recency of hour, day, week, month or year; no extra arguments.",
            ) from None
        instruction = (
            "Conduct thorough research with cited sources on: "
            if name == RESEARCH
            else "Search the web and answer with cited sources: "
        )
        text = instruction + args.query
        if args.recency:
            text += f"\nLimit sources to the last {args.recency}."
        return types.GetPromptResult(
            description="Research request" if name == RESEARCH else "Web search request",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=text),
                )
            ],
        )

    return Server(
        "perplexity-mcp",
        version=__version__,
        lifespan=lifespan,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        on_list_prompts=list_prompts,
        on_get_prompt=get_prompt,
    )


async def main_async(settings: Settings | None = None) -> None:
    config = settings or Settings.from_env()
    server = create_server(config)
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="perplexity-mcp",
                server_version=__version__,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Perplexity search and research MCP server (stdio)"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.parse_args()
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    try:
        config = Settings.from_env()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    try:
        asyncio.run(main_async(config))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
