"""Deterministic loopback-only Sonar substitute for transport tests and MCP Probe.

This deliberately does not model search quality, billing, or provider availability.
No real key is accepted or needed. The server never makes outgoing requests.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

OFFLINE_KEY = "offline-test-only-never-a-real-perplexity-key"


def offline_environment(url: str, **overrides: str) -> dict[str, str]:
    """Never inherit a user's provider credentials or production configuration."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("PERPLEXITY_")}
    env.update(
        PERPLEXITY_API_KEY=OFFLINE_KEY,
        PERPLEXITY_API_URL=url,
        PERPLEXITY_MAX_RETRIES="0",
        PERPLEXITY_SEARCH_TIMEOUT="2",
        PERPLEXITY_RESEARCH_TIMEOUT="2",
    )
    env.update(overrides)
    return env


class FakePerplexity(AbstractContextManager):
    """Record real HTTP requests and dispatch bounded scenarios by query text."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.counts: Counter[str] = Counter()
        self.lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                query = payload["messages"][-1]["content"]
                with owner.lock:
                    owner.requests.append(
                        {
                            "path": self.path,
                            "payload": payload,
                            "authorized": self.headers.get("Authorization")
                            == f"Bearer {OFFLINE_KEY}",
                        }
                    )
                    owner.counts[query] += 1
                    attempt = owner.counts[query]

                status = 200
                headers = {}
                body: Any = {
                    "id": "synthetic-sonar-response",
                    "model": payload["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": f"Synthetic answer: {query} [1]",
                            },
                            "finish_reason": "length" if query == "offline:truncated" else "stop",
                        }
                    ],
                    "citations": ["https://example.org/offline-source"],
                    "search_results": [
                        {
                            "title": "Synthetic source (no live search)",
                            "url": "https://example.org/offline-source",
                            "date": "2026-01-01",
                            "snippet": "Deterministic offline fixture.",
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
                }
                if query.startswith("offline:status:"):
                    status = int(query.rsplit(":", 1)[1])
                    # Deliberately hostile: responses and logs must not leak raw provider bodies.
                    body = {"error": {"message": f"upstream echoed credential: {OFFLINE_KEY}"}}
                elif query == "offline:retry" and attempt == 1:
                    status = 429
                    headers["Retry-After"] = "0"
                    body = {"error": {"message": "Synthetic rate limit"}}
                elif query == "offline:invalid-shape":
                    body = {"unexpected": "response"}
                elif query == "offline:invalid-unicode":
                    body["choices"][0]["message"]["content"] = "bad " + chr(0xD800)
                elif query == "offline:invalid-metadata-unicode":
                    body["usage"]["nested"] = {"annotation": ["bad " + chr(0xD800)]}
                elif query.startswith("offline:delay:"):
                    time.sleep(min(float(query.rsplit(":", 1)[1]), 3.0))

                raw = b"not JSON" if query == "offline:invalid-json" else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                for name, value in headers.items():
                    self.send_header(name, value)
                self.end_headers()
                try:
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    # A timeout/cancellation intentionally closes the client connection.
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1/sonar"

    def __enter__(self) -> FakePerplexity:
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
