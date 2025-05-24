"""Perplexity MCP package."""

__version__ = "1.0.0"

from . import server
import asyncio


def main():
    """Main entry point for the package."""
    asyncio.run(server.main_async())

__all__ = ["main", "server"]
