#!/usr/bin/env python3
"""Run MCP Probe against the installed server with a synthetic loopback provider.

Run from a checkout after installing this project and mcp-probe[full] into the
same Python environment. No API key, network search, or billable call is used.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from fake_perplexity import OFFLINE_KEY, FakePerplexity, offline_environment  # noqa: E402

ACTIVE_SUITES = "lifecycle,jsonrpc,tools,notifications,edge"
CASES = {
    "perplexity_search_web": {
        "query": "Offline search contract check",
        "recency": "week",
        "max_tokens": 64,
    },
    "perplexity_deep_research": {"query": "Offline research contract check", "max_tokens": 64},
}


def write_junit(report: dict, destination: Path) -> None:
    """Use Probe's own formatter for the exact same run captured in JSON."""
    from mcp_probe.reporter import report_junit
    from mcp_probe.types import CheckResult, ProbeReport, Severity, Status, SuiteResult

    typed = ProbeReport(
        probe_version=report["mcp_probe_version"],
        spec_version=report["spec_version"],
        target=report["target"],
        transport=report["transport"],
        timestamp=report["timestamp"],
        duration_ms=report["duration_ms"],
        server_info=report["server_info"],
        capabilities=report["capabilities"],
        incomplete=report["incomplete"],
        mode=report["mode"],
        suites=[
            SuiteResult(
                name=suite["name"],
                checks=[
                    CheckResult(
                        check_id=check["id"],
                        description=check["description"],
                        status=Status(check["status"]),
                        severity=Severity(check["severity"]),
                        duration_ms=check["duration_ms"],
                        details=check.get("details"),
                        error_kind=check.get("error_kind"),
                    )
                    for check in suite["checks"]
                ],
            )
            for suite in report["suites"]
        ],
    )
    destination.write_text(report_junit(typed, strict=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "probe-offline")
    args = parser.parse_args()
    if importlib.util.find_spec("mcp_probe") is None:
        parser.error(
            "Install mcp-probe[full] in this Python environment before running this script."
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases_path = output / "probe-cases.json"
    cases_path.write_text(json.dumps(CASES, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "validation_mode": "offline-synthetic-provider",
        "live_provider_verified": False,
        "limitations": [
            "The loopback substitute verifies HTTP/MCP integration, not real API acceptance, "
            "search quality, or billing.",
            "Active Probe prompt retrieval supplies recency='test'; this invalid enum is "
            "intentionally rejected. Probe prompt coverage is discovery-only; valid and invalid "
            "retrieval are tested by the official SDK suite.",
            "Resources are not advertised, so their suite is not requested. "
            "Notifications that were not emitted remain explicit Probe skips.",
        ],
        "runs": [],
    }
    exit_code = 0
    with FakePerplexity() as provider:
        env = offline_environment(provider.url)
        command = [
            sys.executable,
            "-m",
            "mcp_probe",
            shlex.join([sys.executable, "-m", "perplexity_mcp"]),
            "--cwd",
            str(ROOT),
            "--timeout",
            "8",
            "--run-timeout",
            "60",
            "--strict",
            "--format",
            "json",
        ]
        runs = [
            ("probe-active", ["--active", "--suite", ACTIVE_SUITES, "--cases", str(cases_path)]),
            ("probe-prompts-discovery", ["--suite", "prompts"]),
        ]
        for name, extra in runs:
            destination = output / f"{name}.json"
            destination.unlink(missing_ok=True)
            completed = subprocess.run(
                command + extra + ["--output", str(destination)],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=75,
            )
            # Never reproduce credential-bearing provider bodies, even if a regression logs one.
            stderr = completed.stderr.replace(OFFLINE_KEY, "[REDACTED]")
            (output / f"{name}.stderr.txt").write_text(stderr, encoding="utf-8")
            exit_code = max(exit_code, completed.returncode)
            if not destination.exists():
                print(
                    f"Probe failed before producing {destination.name}: {stderr}", file=sys.stderr
                )
                return completed.returncode or 1
            report = json.loads(destination.read_text(encoding="utf-8"))
            write_junit(report, output / f"{name}.xml")
            provenance["runs"].append(
                {"name": name, "exit_code": completed.returncode, "summary": report["summary"]}
            )
        models = sorted({request["payload"]["model"] for request in provider.requests})
        provenance["loopback_requests"] = len(provider.requests)
        provenance["models_exercised"] = models
        if models != ["sonar", "sonar-deep-research"] or not all(
            r["authorized"] for r in provider.requests
        ):
            print(
                "Both tools must reach the loopback provider with the synthetic credential.",
                file=sys.stderr,
            )
            exit_code = 1
    (output / "probe-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
