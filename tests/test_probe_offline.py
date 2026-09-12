"""Validate the reproducible cross-project MCP Probe workflow and its evidence."""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from fake_perplexity import OFFLINE_KEY

ROOT = Path(__file__).resolve().parents[1]


def test_offline_probe_reports_real_tool_calls_and_honest_scope(tmp_path, monkeypatch):
    pytest.importorskip("mcp_probe", reason="optional cross-project check: install mcp-probe[full]")
    monkeypatch.setenv("PERPLEXITY_API_KEY", "must-never-escape-from-the-parent-environment")
    monkeypatch.setenv("PERPLEXITY_API_URL", "https://invalid.example/do-not-contact")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "probe_offline.py"), "--output-dir", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=150,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert OFFLINE_KEY not in completed.stdout + completed.stderr
    assert (
        "must-never-escape-from-the-parent-environment" not in completed.stdout + completed.stderr
    )
    provenance = json.loads((tmp_path / "probe-provenance.json").read_text())
    assert provenance["validation_mode"] == "offline-synthetic-provider"
    assert provenance["live_provider_verified"] is False
    assert provenance["models_exercised"] == ["sonar", "sonar-deep-research"]
    assert provenance["loopback_requests"] >= 2
    for name, mode in [("probe-active", "active"), ("probe-prompts-discovery", "discovery")]:
        report = json.loads((tmp_path / f"{name}.json").read_text())
        assert report["mode"] == mode
        assert report["incomplete"] is False
        assert report["summary"]["failed"] == report["summary"]["warnings"] == 0
        assert report["summary"]["passed"] > 0
        root = ET.parse(tmp_path / f"{name}.xml").getroot()
        assert root.attrib["failures"] == root.attrib["errors"] == "0"
        assert int(root.attrib["tests"]) == report["summary"]["total"]
    active = json.loads((tmp_path / "probe-active.json").read_text())
    tools = next(suite for suite in active["suites"] if suite["name"] == "tools")
    assert all(check["status"] not in {"FAIL", "WARN"} for check in tools["checks"])
