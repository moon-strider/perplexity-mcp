# Validation — 1.1.0

Executed on 2026-09-12 without a Perplexity API key. All provider responses in the tests and Probe reports are synthetic responses from real loopback HTTP servers. No billable request was made.

## Executed checks

| Check | Result |
|---|---|
| Python 3.11.16 full suite, final code | 197 passed |
| Python 3.12.3 full suite before final metadata-depth regressions | 195 passed; 97% combined statement/branch coverage, including subprocesses |
| Provider/config final regression suite | 182 passed; includes excessive metadata depth |
| Clean wheel install, outside source directory | Real MCP SDK tool/prompt roundtrips pass |
| Modern MCP protocol | Real 2026-07-28 envelopes, discovery, calls, errors and recovery pass |
| Legacy MCP protocol | SDK initialize handshake (2025-11-25), calls, cancellation and recovery pass |
| Ruff lint and formatting | Pass |
| Wheel and sdist; Twine metadata checks | Pass |
| Dependency audit of installed environment | No known vulnerabilities reported by pip-audit at this date |
| MCP Probe | 44 checks: 27 pass, 12 explicit skips, 5 informational; no failures or warnings |

The installed-wheel gate launches the server from a temporary working directory using a separate virtual environment. It does not use an editable install or the source tree for server imports.

CI additionally covers Python 3.13 and Docker image build/start. Docker is not installed in the local execution environment; its result must be read from the actual GitHub Actions run, not inferred from the unit tests. Package registry publication and live API verification were not performed. The dependency audit checked 68 published distributions; the local project itself was skipped because version 1.1.0 was not on PyPI. Its source is covered by the review and tests above.

## Independent protocol evidence

- [Active tool/protocol report](evidence/probe/probe-active.json)
- [Active JUnit](evidence/probe/probe-active.xml)
- [Prompt discovery report](evidence/probe/probe-prompts-discovery.json)
- [Prompt discovery JUnit](evidence/probe/probe-prompts-discovery.xml)
- [Provenance and scope](evidence/probe/probe-provenance.json)

Probe is pinned to moon-strider/mcp-probe commit `21d435e8ab68b216a98a14a5f147730912edef7c`. Its generic active prompt retrieval generates recency=`test`, which this server correctly rejects. Prompt retrieval is therefore tested with valid/invalid inputs using the official SDK, while Probe runs prompt discovery. Unsupported resources and unobserved notifications are not claimed as tested capabilities. Both tools demonstrably reach the loopback provider.

## Failure scenarios covered

Tests inspect the actual HTTP payload, including auth, integer token limits, nested search options, optional filters, and model selection. They cover concurrent tool calls; timeout and cancellation followed by a healthy session; retry counts and Retry-After under a total deadline; refusal to replay disconnected or timed-out work; redirects; bounded bodies; invalid JSON, overflowing numbers, escaped surrogates and excessive nested metadata; nullable source/usage fields; citation numbering and truncation; and key redaction from returned metadata and errors.

An independent source/blackbox review found and led to regression fixes for the new SDK callback API, nullable upstream metadata, overflowing JSON numbers, prompt validation values leaking in SDK exception logs, Unicode responses crashing the stdio writer, and metadata nesting exceeding the serializer's depth limit.

MCP SDK transport limitation: malformed Unicode JSON frames can be rejected before application handlers and silently discarded. A raw-protocol regression proves a subsequent valid request still works; this project does not claim an error envelope for every malformed byte sequence.

## Remaining live check

With a real key, run one normal search and one research request through an MCP client. Record the model, answer, cited URLs, usage, finish reason and latency; check source correspondence manually. This verifies production payload acceptance and account/model availability. The offline suite cannot establish search quality, citation truth, billing, current model entitlements, or cancellation of already-running provider work.
