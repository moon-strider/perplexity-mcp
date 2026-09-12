"""Validated, immutable server configuration.

Configuration failures identify the variable without printing its value. An API
key is deliberately excluded from the dataclass representation.
"""

from __future__ import annotations

import ipaddress
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit


class ConfigurationError(ValueError):
    """An invalid or missing environment setting."""


def _invalid(name: str, requirement: str) -> ConfigurationError:
    return ConfigurationError(f"{name} {requirement}")


def _integer(name: str, value: object, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise _invalid(name, f"must be an integer from {minimum} to {maximum}")
    if isinstance(value, str):
        if not re.fullmatch(r"[0-9]+", value):
            raise _invalid(name, f"must be an integer from {minimum} to {maximum}")
        try:
            value = int(value)
        except ValueError:
            raise _invalid(name, f"must be an integer from {minimum} to {maximum}") from None
    if not isinstance(value, int) or not minimum <= value <= maximum:
        raise _invalid(name, f"must be an integer from {minimum} to {maximum}")
    return value


def _seconds(name: str, value: object, maximum: float) -> float:
    if isinstance(value, bool):
        raise _invalid(name, f"must be a finite number from 0.001 to {maximum:g}")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (ValueError, TypeError, OverflowError):
        raise _invalid(name, f"must be a finite number from 0.001 to {maximum:g}") from None
    if not math.isfinite(number) or not 0.001 <= number <= maximum:
        raise _invalid(name, f"must be a finite number from 0.001 to {maximum:g}")
    return number


def _endpoint(value: str) -> str:
    requirement = (
        "must be an HTTPS URL (HTTP is allowed only for loopback), "
        "without credentials, query, or fragment"
    )
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise _invalid("PERPLEXITY_API_URL", requirement)
    if any(character.isspace() or ord(character) < 32 for character in value):
        raise _invalid("PERPLEXITY_API_URL", requirement)
    if "?" in value or "#" in value or "\\" in value:
        raise _invalid("PERPLEXITY_API_URL", requirement)
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
        if not host or parsed.username is not None or parsed.password is not None:
            raise ValueError
        if port is not None and not 1 <= port <= 65535:
            raise ValueError
        is_loopback = host.lower() == "localhost"
        try:
            is_loopback = is_loopback or ipaddress.ip_address(host).is_loopback
        except ValueError:
            pass
        if parsed.scheme != "https" and not (parsed.scheme == "http" and is_loopback):
            raise ValueError
    except ValueError:
        raise _invalid("PERPLEXITY_API_URL", requirement) from None
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str = field(repr=False)
    api_url: str = "https://api.perplexity.ai/v1/sonar"
    model: str = "sonar"
    research_model: str = "sonar-deep-research"
    search_max_tokens: int = 8192
    research_max_tokens: int = 8192
    search_timeout: float = 60.0
    research_timeout: float = 600.0
    max_retries: int = 2
    response_limit: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        if (
            not isinstance(self.api_key, str)
            or not 1 <= len(self.api_key) <= 4096
            or not self.api_key.isascii()
            or any(not character.isprintable() or character.isspace() for character in self.api_key)
        ):
            raise _invalid(
                "PERPLEXITY_API_KEY",
                "is required and must contain only non-whitespace ASCII characters",
            )
        _endpoint(self.api_url)
        for name, value in (
            ("PERPLEXITY_MODEL", self.model),
            ("PERPLEXITY_RESEARCH_MODEL", self.research_model),
        ):
            if not isinstance(value, str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", value
            ):
                raise _invalid(
                    name, "must be a nonempty model identifier of at most 128 characters"
                )
        for name, attribute in (
            ("PERPLEXITY_SEARCH_MAX_TOKENS", "search_max_tokens"),
            ("PERPLEXITY_RESEARCH_MAX_TOKENS", "research_max_tokens"),
        ):
            object.__setattr__(self, attribute, _integer(name, getattr(self, attribute), 1, 65536))
        object.__setattr__(
            self, "search_timeout", _seconds("PERPLEXITY_SEARCH_TIMEOUT", self.search_timeout, 3600)
        )
        object.__setattr__(
            self,
            "research_timeout",
            _seconds("PERPLEXITY_RESEARCH_TIMEOUT", self.research_timeout, 7200),
        )
        object.__setattr__(
            self, "max_retries", _integer("PERPLEXITY_MAX_RETRIES", self.max_retries, 0, 5)
        )
        object.__setattr__(
            self,
            "response_limit",
            _integer("response_limit", self.response_limit, 1024, 32 * 1024 * 1024),
        )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ
        return cls(
            api_key=env.get("PERPLEXITY_API_KEY", ""),
            api_url=env.get("PERPLEXITY_API_URL", "https://api.perplexity.ai/v1/sonar"),
            model=env.get("PERPLEXITY_MODEL", "sonar"),
            research_model=env.get("PERPLEXITY_RESEARCH_MODEL", "sonar-deep-research"),
            search_max_tokens=_integer(
                "PERPLEXITY_SEARCH_MAX_TOKENS",
                env.get("PERPLEXITY_SEARCH_MAX_TOKENS", "8192"),
                1,
                65536,
            ),
            research_max_tokens=_integer(
                "PERPLEXITY_RESEARCH_MAX_TOKENS",
                env.get("PERPLEXITY_RESEARCH_MAX_TOKENS", "8192"),
                1,
                65536,
            ),
            search_timeout=_seconds(
                "PERPLEXITY_SEARCH_TIMEOUT", env.get("PERPLEXITY_SEARCH_TIMEOUT", "60"), 3600
            ),
            research_timeout=_seconds(
                "PERPLEXITY_RESEARCH_TIMEOUT", env.get("PERPLEXITY_RESEARCH_TIMEOUT", "600"), 7200
            ),
            max_retries=_integer(
                "PERPLEXITY_MAX_RETRIES", env.get("PERPLEXITY_MAX_RETRIES", "2"), 0, 5
            ),
        )
