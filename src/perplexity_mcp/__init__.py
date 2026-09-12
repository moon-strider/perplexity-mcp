"""Perplexity search and research over MCP."""

__version__ = "1.1.0"


def main() -> None:
    """Console entry point; import runtime only when invoked."""
    from .server import main as run

    run()
