# Perplexity MCP

A small Python MCP server for cited web answers and longer research requests through Perplexity's Sonar API. Works over **stdio** with two tools, validated arguments, bounded requests, and structured results containing sources and usage.

[![CI](https://github.com/moon-strider/perplexity-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/moon-strider/perplexity-mcp/actions/workflows/ci.yml)

## Install and connect

Requires Python 3.11+ and a Perplexity API key. The package and executable are named **`perplexity-mcp-ultra`**.

```sh
uv tool install 'git+https://github.com/moon-strider/perplexity-mcp.git'
export PERPLEXITY_API_KEY='your-key'
perplexity-mcp-ultra
```

The process speaks MCP on stdin/stdout; starting it directly waits for an MCP client. Diagnostics go to stderr. `--help` and `--version` work without a key.

A complete configuration for clients using `mcpServers`:

```json
{
  "mcpServers": {
    "perplexity": {
      "command": "perplexity-mcp-ultra",
      "env": {
        "PERPLEXITY_API_KEY": "your-key"
      }
    }
  }
}
```

Desktop clients may need the executable's absolute path; `command -v perplexity-mcp-ultra` shows it on Unix. Use the client's secret/environment settings when available.

## Tools

| Tool | Default model | Use |
|---|---|---|
| `perplexity_search_web` | `sonar` | A cited answer to a web search question |
| `perplexity_deep_research` | `sonar-deep-research` | A longer research request; allow several minutes |

Both accept:

- `query`: nonblank string, at most 32,000 characters.
- `recency`: `hour`, `day`, `week`, `month`, `year`, or `null`. Omit it for **no time restriction**.
- `max_tokens`: integer from 1 to 65,536, or `null` to use the configured default. This limits answer generation, not the model's context window. Provider/model limits still apply.

Unknown arguments and invalid types are rejected before any HTTP call. These examples show MCP tool calls, not shell commands:

```json
{"name":"perplexity_search_web","arguments":{"query":"What changed in Python packaging this week?","recency":"week","max_tokens":2048}}
```

```json
{"name":"perplexity_deep_research","arguments":{"query":"Compare strategies for evaluating long-running software agents, with primary sources.","max_tokens":8192}}
```

The same names are available as MCP prompts accepting `query` and optional `recency`. Getting a prompt only creates an instruction; it makes no paid API call.

## Results

Each successful call returns readable text and `structuredContent` with:

- `answer`, `citations`, and `search_results` (including available source metadata);
- `model`, provider `usage`, and `finish_reason`;
- `truncated`, `latency_seconds`, and `attempts`.

Citation order and duplicates are preserved so `[1]`, `[2]`, etc. still refer to the provider's answer. Search results do not create invented citation numbers. A token-limited answer is marked incomplete. Missing optional usage/source fields are represented by empty containers; reported usage/cost is whatever the provider supplied, not an estimate.

Tool failures have `isError: true` and `structuredContent.error` with `code`, `message`, `retryable`, and an HTTP `status` when available. Provider response bodies, credentials, and user queries are not copied into diagnostic errors.

## Configuration

| Environment variable | Default | Meaning |
|---|---|---|
| `PERPLEXITY_API_KEY` | Required | Perplexity credential |
| `PERPLEXITY_API_URL` | `https://api.perplexity.ai/v1/sonar` | Full completion endpoint |
| `PERPLEXITY_MODEL` | `sonar` | Search model |
| `PERPLEXITY_RESEARCH_MODEL` | `sonar-deep-research` | Research model |
| `PERPLEXITY_SEARCH_MAX_TOKENS` | `8192` | Default answer token limit |
| `PERPLEXITY_RESEARCH_MAX_TOKENS` | `8192` | Default research answer token limit |
| `PERPLEXITY_SEARCH_TIMEOUT` | `60` | Total seconds, including retries |
| `PERPLEXITY_RESEARCH_TIMEOUT` | `600` | Total seconds, including retries |
| `PERPLEXITY_MAX_RETRIES` | `2` | Extra attempts, allowed range 0–5 |

HTTPS is required for custom endpoints, except loopback HTTP used by local tests. The administrator controls this endpoint; credentials are sent to it. Redirects are not followed. The response body is bounded at 8 MiB and compressed responses are rejected to bound decoded memory use.

Only HTTP 429, 502, 503, and 504 can be retried, within the original request deadline. `Retry-After` is respected; if it cannot fit the remaining deadline, no early retry is made. Timeouts and disconnected requests are **not** automatically replayed, because the provider may already have processed/billed them. Transient HTTP retries may also incur provider charges; set retries to `0` for a single attempt. This server does not promise a dollar-denominated budget or exactly-once billing.

Configure the MCP client's own timeout to exceed the research timeout. Cancelling a call stops local waiting; this synchronous API cannot guarantee cancellation of already-running provider work.

Perplexity also maintains an [official MCP server](https://docs.perplexity.ai/docs/getting-started/integrations/mcp-server). This project is a focused Python implementation of the two Sonar tools. The [Sonar API reference](https://docs.perplexity.ai/api-reference/sonar-post) currently groups this API under Legacy API; no shutdown claim is made here. Other Perplexity API families are outside this package.

## Docker

```sh
docker build -t perplexity-mcp .
docker run --rm -i -e PERPLEXITY_API_KEY perplexity-mcp
```

Use `-i` without a TTY. The image runs the installed executable as an unprivileged user. No API key is embedded in the image.

## Development and offline verification

```sh
uv sync --frozen --group dev
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pytest --cov=perplexity_mcp --cov-branch --cov-report=term-missing
uv build
uv run --frozen twine check dist/*
```

The suite starts real loopback HTTP servers and real MCP stdio processes. It checks success, malformed/oversized responses, validation, retries, deadlines, cancellation, concurrent calls, source preservation, credential redaction, and lifecycle cleanup without a Perplexity key. Tests never call the public Perplexity endpoint.

To independently check the protocol with [mcp-probe](https://github.com/moon-strider/mcp-probe):

```sh
uv pip install 'mcp-probe[full] @ git+https://github.com/moon-strider/mcp-probe.git@21d435e8ab68b216a98a14a5f147730912edef7c'
.venv/bin/python scripts/probe_offline.py --output-dir reports/probe
```

The script uses an explicitly synthetic loopback provider and exports JSON/JUnit reports. Its active suites cover tool calls and protocol behavior; prompt discovery is checked separately because Probe's generic prompt generator does not know recency's allowed values. Valid and invalid prompt calls are exercised by the SDK tests.

See [validation evidence](docs/validation.md) for executed checks and limitations. **No live Perplexity request has been verified for this revision.** Offline success does not establish model availability, account entitlements, real billing, citation truth, or current production acceptance of the payload. With a key, the remaining smoke is one search and one bounded research call through an MCP client, recording model, sources, usage and latency.

## License

MIT.
