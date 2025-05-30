# perplexity-mcp MCP server

A Model Context Protocol (MCP) server that provides web search functionality using [Perplexity AI's](https://www.perplexity.ai/) API. Works with the [Anthropic](https://www.anthropic.com/news/model-context-protocol) Claude desktop client.

## Examples

Use prompts like:
- "Search the web to find out what's new at Anthropic in the past week."
- "Do deep research on the latest advances in quantum computing in the past month."

## Components

### Prompts

The server provides two prompts:

- **perplexity_search_web**: Quick web search using Perplexity AI
  - Required "query" argument for the search query
  - Optional "recency" argument to filter results by time period:
    - 'day': last 24 hours
    - 'week': last 7 days
    - 'month': last 30 days (default)
    - 'year': last 365 days
  - Uses Perplexity's API to perform web searches

- **perplexity_deep_research**: Comprehensive deep research using Perplexity AI
  - Required "query" argument for the research topic
  - Optional "recency" argument to filter results by time period (same options as above)
  - Uses Perplexity's sonar-deep-research model with 65,536 max tokens
  - Provides exhaustive analysis with detailed insights backed by evidence

### Tools

The server implements two tools:

- **perplexity_search_web**: Quick web search using Perplexity AI
  - Takes "query" as a required string argument
  - Optional "recency" parameter to filter results (day/week/month/year)
  - Returns concise search results from Perplexity's API
  - Best for quick lookups and current information

- **perplexity_deep_research**: Comprehensive deep research using Perplexity AI
  - Takes "query" as a required string argument
  - Optional "recency" parameter to filter results (day/week/month/year)
  - Returns extensive, detailed research with citations
  - Best for in-depth analysis, comprehensive reports, and thorough investigations

## Installation

### Requires [UV](https://github.com/astral-sh/uv) (Fast Python package and project manager)

If uv isn't installed.

```bash
# Using Homebrew on macOS
brew install uv
```

or

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Environment Variables

The following environment variable is required in your claude_desktop_config.json. You can obtain an API key from [Perplexity](https://perplexity.ai)

- `PERPLEXITY_API_KEY`: Your Perplexity AI API key

Optional environment variables:

- `PERPLEXITY_MODEL`: The Perplexity model to use (defaults to "sonar" if not specified)

  Available models:

  - `sonar-deep-research`: 128k context - Enhanced research capabilities
  - `sonar-reasoning-pro`: 128k context - Advanced reasoning with professional focus
  - `sonar-reasoning`: 128k context - Enhanced reasoning capabilities
  - `sonar-pro`: 200k context - Professional grade model
  - `sonar`: 128k context - Default model

And updated list of models is avaiable (here)[https://docs.perplexity.ai/guides/model-cards]

### Cursor & Claude Desktop Installation

Add this tool as a mcp server by editing the Cursor/Claude config file.

> There is sometimes a problem with uvx scope in Claude Desktop, so if logs show that uvx is not found, try running `which uvx` from the terminal that has access to uvx and then specity the full path to the executable in "command" of the mcp config.

```json
  "perplexity-mcp": {
    "env": {
      "PERPLEXITY_API_KEY": "XXXXXXXXXXXXXXXXXXXX",
      "PERPLEXITY_MODEL": "sonar"
    },
    "command": "uvx", # or full path like: /opt/homebrew/Caskroom/miniconda/base/bin/uvx
    "args": [
      "--from",
      "git+https://github.com/moon-strider/perplexity-mcp",
      "perplexity-mcp-ultra"
    ]
  }
```

#### Cursor
- On MacOS: `/Users/your-username/.cursor/mcp.json`
- On Windows: `C:\Users\your-username\.cursor\mcp.json`

If everything is working correctly, you should now be able to call the tool from Cursor.
<img width="800" alt="mcp_screenshot" src="https://github.com/user-attachments/assets/4b59774f-646c-41b3-9886-5cfd4c4ca051" width=600>

#### Claude Desktop
- On MacOS: `~/Library/Application\ Support/Claude/claude_desktop_config.json`
- On Windows: `%APPDATA%/Claude/claude_desktop_config.json`

To verify the server is working. Open the Claude client and use a prompt like "search the web for news about openai in the past week". You should see an alert box open to confirm tool usage. Click "Allow for this chat".

  <img width="600" alt="mcp_screenshot" src="https://github.com/user-attachments/assets/922d8f6a-8c9a-4978-8be6-788e70b4d049" />
