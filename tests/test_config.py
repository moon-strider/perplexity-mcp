from dataclasses import FrozenInstanceError

import pytest

from perplexity_mcp.config import ConfigurationError, Settings

KEY = "pplx-offline-secret"


def test_defaults_and_secret_representation():
    settings = Settings.from_env({"PERPLEXITY_API_KEY": KEY})
    assert settings.api_url == "https://api.perplexity.ai/v1/sonar"
    assert settings.model == "sonar"
    assert settings.research_model == "sonar-deep-research"
    assert settings.search_max_tokens == settings.research_max_tokens == 8192
    assert settings.search_timeout == 60
    assert settings.research_timeout == 600
    assert settings.max_retries == 2
    assert settings.response_limit == 8 * 1024 * 1024
    assert KEY not in repr(settings)
    with pytest.raises(FrozenInstanceError):
        settings.model = "changed"


def test_explicit_environment_overrides(monkeypatch):
    values = {
        "PERPLEXITY_API_KEY": KEY,
        "PERPLEXITY_API_URL": "http://127.0.0.1:1234/sonar",
        "PERPLEXITY_MODEL": "sonar-pro",
        "PERPLEXITY_RESEARCH_MODEL": "sonar-deep-research",
        "PERPLEXITY_SEARCH_MAX_TOKENS": "64",
        "PERPLEXITY_RESEARCH_MAX_TOKENS": "65536",
        "PERPLEXITY_SEARCH_TIMEOUT": "12.5",
        "PERPLEXITY_RESEARCH_TIMEOUT": "90",
        "PERPLEXITY_MAX_RETRIES": "0",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    settings = Settings.from_env()
    assert settings.api_url == values["PERPLEXITY_API_URL"]
    assert settings.model == "sonar-pro"
    assert settings.search_max_tokens == 64
    assert settings.research_max_tokens == 65536
    assert settings.search_timeout == 12.5
    assert settings.research_timeout == 90
    assert settings.max_retries == 0


@pytest.mark.parametrize(
    "key", ["", " ", "bad key", "key\n", "key\r", "key\x00", "ключ", "a" * 4097, None]
)
def test_invalid_keys_never_echo_values(key):
    with pytest.raises(ConfigurationError) as caught:
        Settings(api_key=key)
    assert str(caught.value).startswith("PERPLEXITY_API_KEY ")
    assert "bad key" not in str(caught.value)


def test_missing_key():
    with pytest.raises(ConfigurationError, match="PERPLEXITY_API_KEY"):
        Settings.from_env({})


@pytest.mark.parametrize(
    "url",
    [
        "https://api.perplexity.ai/v1/sonar",
        "https://example.com/custom/v1/sonar",
        "https://example.com:8443/",
        "http://127.0.0.1:8080/v1/sonar",
        "http://127.0.0.2:8080/v1/sonar",
        "http://localhost:8080/",
        "http://[::1]:8080/",
    ],
)
def test_allowed_endpoints(url):
    assert Settings(api_key=KEY, api_url=url).api_url == url


@pytest.mark.parametrize(
    "url",
    [
        "",
        "api.perplexity.ai",
        "ftp://localhost/a",
        "http://api.perplexity.ai/a",
        "http://localhost.attacker.example/a",
        "http://192.168.1.1/a",
        "http://0.0.0.0/a",
        "http://[::]/a",
        "https:///a",
        "https://user:secret@example.com/a",
        "https://user@example.com/a",
        "https://example.com/?token=secret",
        "https://example.com/#secret",
        "https://example.com/?",
        "https://example.com/#",
        "https://example.com:0/a",
        "https://example.com:65536/a",
        "https://example.com:bad/a",
        "https://[::1/a",
        "https://example.com/\nsecret",
        "https://example.com/ secret",
        "https://example.com\\secret",
        None,
    ],
)
def test_invalid_endpoints_are_safe(url):
    with pytest.raises(ConfigurationError) as caught:
        Settings(api_key=KEY, api_url=url)
    assert "PERPLEXITY_API_URL" in str(caught.value)
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("field", ["model", "research_model"])
@pytest.mark.parametrize(
    "value", ["", " ", "invalid model", "a\n", "a" * 129, "-a", "кириллица", None]
)
def test_invalid_models(field, value):
    with pytest.raises(ConfigurationError):
        Settings(api_key=KEY, **{field: value})


@pytest.mark.parametrize("field", ["search_max_tokens", "research_max_tokens"])
@pytest.mark.parametrize("value", [0, -1, 65537, True, 3.0, "3.0", "nan", " 3", None, "9" * 5000])
def test_invalid_token_limits(field, value):
    with pytest.raises(ConfigurationError):
        Settings(api_key=KEY, **{field: value})


@pytest.mark.parametrize("field", ["search_timeout", "research_timeout"])
@pytest.mark.parametrize(
    "value", [0, -1, 0.0001, 7201, True, float("nan"), float("inf"), "-inf", "secret", None]
)
def test_invalid_timeouts(field, value):
    with pytest.raises(ConfigurationError) as caught:
        Settings(api_key=KEY, **{field: value})
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("value", [-1, 6, True, 1.1, "1.0", None])
def test_invalid_retries(value):
    with pytest.raises(ConfigurationError):
        Settings(api_key=KEY, max_retries=value)


@pytest.mark.parametrize("value", [0, 1023, 32 * 1024 * 1024 + 1, True])
def test_invalid_response_limit(value):
    with pytest.raises(ConfigurationError):
        Settings(api_key=KEY, response_limit=value)


@pytest.mark.parametrize(
    "name,value",
    [
        ("PERPLEXITY_SEARCH_MAX_TOKENS", "1.1"),
        ("PERPLEXITY_RESEARCH_MAX_TOKENS", "65537"),
        ("PERPLEXITY_SEARCH_TIMEOUT", "nan"),
        ("PERPLEXITY_RESEARCH_TIMEOUT", "Infinity"),
        ("PERPLEXITY_MAX_RETRIES", "6"),
    ],
)
def test_invalid_environment_values(name, value):
    with pytest.raises(ConfigurationError, match=name):
        Settings.from_env({"PERPLEXITY_API_KEY": KEY, name: value})


def test_numeric_boundaries():
    settings = Settings(
        api_key=KEY,
        search_max_tokens=1,
        research_max_tokens=65536,
        search_timeout=0.001,
        research_timeout=7200,
        max_retries=5,
    )
    assert settings.search_max_tokens == 1
    assert settings.research_max_tokens == 65536
    assert settings.max_retries == 5
