import asyncio
import aiohttp
import sys
import logging
import os

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
from perplexity_mcp import __version__

server = Server("perplexity-mcp")


@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="perplexity_search_web",
            description="Search the web using Perplexity AI and filter results by recency if required",
            arguments=[
                types.PromptArgument(
                    name="query",
                    description="The search query to find information about",
                    required=True,
                ),
                types.PromptArgument(
                    name="recency",
                    description="Filter results by how recent they are. Options: 'day' (last 24h), 'week' (last 7 days), 'month' (last 30 days), 'year' (last 365 days). Optional argument.",
                    required=False,
                ),
            ],
        )
    ]


@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    if name != "perplexity_search_web":
        raise ValueError(f"Unknown prompt: {name}")

    query = (arguments or {}).get("query", "")
    recency = (arguments or {}).get("recency", None)
    msg_composed = [
        types.PromptMessage(
            role="user",
            content=types.TextContent(
                type="text",
                text=f"Find recent information about: {query}",
            ),
        ),
    ]
    if recency:
        msg_composed.append(
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"Only include results from the last {recency}",
                ),
            ),
        )
    return types.GetPromptResult(
        description=f"Search the web for information about: {query}",
        messages=msg_composed,
    )


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="perplexity_search_web",
            description="Search the web using Perplexity AI with recency filtering",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "recency": {
                        "type": "string",
                        "enum": ["day", "week", "month", "year"],
                        "default": "month",
                    },
                },
                "required": ["query"],
            },
        )
    ]


@server.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if name == "perplexity_search_web":
        query = arguments["query"]
        recency = arguments.get("recency", "month")
        result = await call_perplexity(query, recency)
        return [types.TextContent(type="text", text=str(result))]
    raise ValueError(f"Tool not found: {name}")


async def call_perplexity(query: str, recency: str | None = None) -> str:

    url = "https://api.perplexity.ai/chat/completions"

    model = os.getenv("PERPLEXITY_MODEL", "sonar-pro")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Be precise in your responses, never make assumptions you cannot back up."},
            {"role": "user", "content": query},
        ],
        "max_tokens": "4096",
        "temperature": 0.2,
        "top_p": 0.9,
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": recency,
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1,
        "return_citations": True,
        "search_context_size": "high",
    }

    headers = {
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
            content = data["choices"][0]["message"]["content"]
            
            if "citations" in data:
                citations = data["citations"]
                formatted_citations = "\n\nCitations:\n" + "\n".join(f"[{i+1}] {url}" for i, url in enumerate(citations))
                return content + formatted_citations
            
            return content


async def main_async():
    API_KEY = os.getenv("PERPLEXITY_API_KEY")
    if not API_KEY:
        raise ValueError("PERPLEXITY_API_KEY environment variable is required")

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


def main():
    logging.basicConfig(level=logging.INFO)

    API_KEY = os.getenv("PERPLEXITY_API_KEY")
    if not API_KEY:
        print(
            "Error: PERPLEXITY_API_KEY environment variable is required",
            file=sys.stderr,
        )
        sys.exit(1)

    model = os.getenv("PERPLEXITY_MODEL", "sonar")
    logging.info(f"Using Perplexity AI model: {model}")
    
    available_models = {
        "sonar-deep-research": "128k context - Enhanced research capabilities",
        "sonar-reasoning-pro": "128k context - Advanced reasoning with professional focus",
        "sonar-reasoning": "128k context - Enhanced reasoning capabilities",
        "sonar-pro": "200k context - Professional grade model",
        "sonar": "128k context - Default model",
        "r1-1776": "128k context - Alternative architecture"
    }
    
    logging.info("Available Perplexity models (set with PERPLEXITY_MODEL environment variable):")
    for model_name, description in available_models.items():
        marker = "→" if model_name == model else " "
        logging.info(f" {marker} {model_name}: {description}")

    asyncio.run(main_async())


if __name__ == "__main__":
    main()
