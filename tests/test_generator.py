"""Tests for generate_regression_test() and create_regression_pr()."""

import json
import os
import pytest

import trial.config as cfg
from trial import generate_regression_test, create_regression_pr
from trial.generator import _get_github_repo, _safe_id
from trial.providers.base import BaseProvider
from trial.tools import ToolCall


USER_MSG = "Give me a pasta recipe"
RESPONSE = "Error: upstream timeout"

ANALYSIS_JSON = """{
  "failure_description": "Agent crashed with an upstream timeout instead of returning a recipe",
  "judge_criterion": "Agent returns a complete pasta recipe without errors",
  "include_no_errors": true,
  "include_latency": false,
  "latency_threshold": null
}"""

ANALYSIS_JSON_WITH_LATENCY = """{
  "failure_description": "Agent was very slow",
  "judge_criterion": "Agent responds quickly with a complete recipe",
  "include_no_errors": false,
  "include_latency": true,
  "latency_threshold": 5.0
}"""


class FakeProvider(BaseProvider):
    def __init__(self, response: str = ANALYSIS_JSON):
        self._response = response

    def complete(self, system: str, user: str) -> str:
        return self._response


# --- basic generation ---

def test_returns_string():
    cfg._default_provider = FakeProvider()
    try:
        code = generate_regression_test(USER_MSG, RESPONSE)
        assert isinstance(code, str)
        assert len(code) > 0
    finally:
        cfg._default_provider = None


def test_contains_user_message():
    cfg._default_provider = FakeProvider()
    try:
        code = generate_regression_test(USER_MSG, RESPONSE)
        assert USER_MSG in code
    finally:
        cfg._default_provider = None


def test_contains_judge_criterion():
    cfg._default_provider = FakeProvider()
    try:
        code = generate_regression_test(USER_MSG, RESPONSE)
        assert "Agent returns a complete pasta recipe without errors" in code
    finally:
        cfg._default_provider = None


def test_contains_failure_description_in_comment():
    cfg._default_provider = FakeProvider()
    try:
        code = generate_regression_test(USER_MSG, RESPONSE)
        assert "Agent crashed with an upstream timeout" in code
    finally:
        cfg._default_provider = None


def test_contains_from_execution():
    cfg._default_provider = FakeProvider()
    try:
        code = generate_regression_test(USER_MSG, RESPONSE)
        assert "Trial.from_execution" in code
    finally:
        cfg._default_provider = None


def test_contains_assert_passed():
    cfg._default_provider = FakeProvider()
    try:
        code = generate_regression_test(USER_MSG, RESPONSE)
        assert "assert_passed" in code
    finally:
        cfg._default_provider = None


# --- session_id ---

def test_session_id_in_test_function_name():
    cfg._default_provider = FakeProvider()
    try:
        code = generate_regression_test(USER_MSG, RESPONSE, session_id="abc-123")
        assert "def test_regression_abc_123" in code
    finally:
        cfg._default_provider = None


def test_session_id_in_header_comment():
    cfg._default_provider = FakeProvider()
    try:
        code = generate_regression_test(USER_MSG, RESPONSE, session_id="abc-123")
        assert "Session: abc-123" in code
    finally:
        cfg._default_provider = None


def test_no_session_id_uses_fallback_name():
    cfg._default_provider = FakeProvider()
    try:
        code = generate_regression_test(USER_MSG, RESPONSE)
        assert "def test_regression_session" in code
    finally:
        cfg._default_provider = None


# --- deterministic checks in output ---

def test_no_errors_included_when_flagged():
    cfg._default_provider = FakeProvider()  # ANALYSIS_JSON has include_no_errors=true
    try:
        code = generate_regression_test(USER_MSG, RESPONSE)
        assert ".no_errors()" in code
    finally:
        cfg._default_provider = None


def test_no_errors_excluded_when_not_flagged():
    cfg._default_provider = FakeProvider(ANALYSIS_JSON_WITH_LATENCY)
    try:
        code = generate_regression_test(USER_MSG, RESPONSE)
        assert ".no_errors()" not in code
    finally:
        cfg._default_provider = None


def test_latency_check_included_when_flagged():
    cfg._default_provider = FakeProvider(ANALYSIS_JSON_WITH_LATENCY)
    try:
        code = generate_regression_test(USER_MSG, RESPONSE)
        assert ".completes_within(5.0)" in code
    finally:
        cfg._default_provider = None


def test_latency_check_excluded_when_not_flagged():
    cfg._default_provider = FakeProvider()  # include_latency=false
    try:
        code = generate_regression_test(USER_MSG, RESPONSE)
        assert "completes_within" not in code
    finally:
        cfg._default_provider = None


# --- output_path ---

def test_output_path_writes_file(tmp_path):
    cfg._default_provider = FakeProvider()
    path = str(tmp_path / "test_regression.py")
    try:
        code = generate_regression_test(USER_MSG, RESPONSE, output_path=path)
        assert os.path.exists(path)
        with open(path) as f:
            assert f.read() == code
    finally:
        cfg._default_provider = None


def test_no_output_path_writes_no_file(tmp_path):
    cfg._default_provider = FakeProvider()
    try:
        generate_regression_test(USER_MSG, RESPONSE)
        assert list(tmp_path.iterdir()) == []
    finally:
        cfg._default_provider = None


# --- error handling ---

def test_malformed_json_raises():
    cfg._default_provider = FakeProvider("not json at all")
    try:
        with pytest.raises(ValueError, match="malformed JSON"):
            generate_regression_test(USER_MSG, RESPONSE)
    finally:
        cfg._default_provider = None


def test_missing_fields_raises():
    cfg._default_provider = FakeProvider('{"failure_description": "oops"}')
    try:
        with pytest.raises(ValueError, match="missing fields"):
            generate_regression_test(USER_MSG, RESPONSE)
    finally:
        cfg._default_provider = None


def test_no_provider_configured_raises():
    cfg._default_provider = None
    with pytest.raises(RuntimeError, match="No provider configured"):
        generate_regression_test(USER_MSG, RESPONSE)


# --- per-call provider override ---

def test_provider_override():
    provider = FakeProvider()
    # No global provider configured — should still work with explicit provider
    cfg._default_provider = None
    code = generate_regression_test(USER_MSG, RESPONSE, provider=provider)
    assert "Trial.from_execution" in code


# --- tool_calls passed through ---

def test_tool_calls_accepted():
    cfg._default_provider = FakeProvider()
    tool_calls = [ToolCall(name="search_recipe", input={"query": "pasta"})]
    try:
        code = generate_regression_test(USER_MSG, RESPONSE, tool_calls=tool_calls)
        assert isinstance(code, str)
    finally:
        cfg._default_provider = None


# --- _safe_id helper ---

def test_safe_id_none():
    assert _safe_id(None) == "session"

def test_safe_id_alphanumeric():
    assert _safe_id("abc123") == "abc123"

def test_safe_id_hyphens_and_dots():
    assert _safe_id("sess-8821.prod") == "sess_8821_prod"


# --- _get_github_repo helper ---

def test_get_github_repo_https(monkeypatch):
    import subprocess

    class FakeResult:
        stdout = "https://github.com/acme/my-repo.git\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FakeResult())
    owner, repo = _get_github_repo(".")
    assert owner == "acme"
    assert repo == "my-repo"


def test_get_github_repo_ssh(monkeypatch):
    import subprocess

    class FakeResult:
        stdout = "git@github.com:acme/my-repo.git\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FakeResult())
    owner, repo = _get_github_repo(".")
    assert owner == "acme"
    assert repo == "my-repo"


def test_get_github_repo_unknown_format_raises(monkeypatch):
    import subprocess

    class FakeResult:
        stdout = "https://gitlab.com/acme/repo.git\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FakeResult())
    with pytest.raises(ValueError, match="Could not parse"):
        _get_github_repo(".")


# --- create_regression_pr ---

def test_create_regression_pr_missing_token():
    """Passing an empty token still calls through — the API will reject it.
    We just verify the function exists and accepts the right arguments."""
    # This test confirms the signature is correct without making real calls
    import inspect
    sig = inspect.signature(create_regression_pr)
    assert "github_token" in sig.parameters
    assert "session_id" in sig.parameters
    assert "repo_path" in sig.parameters
    assert "base_branch" in sig.parameters


def test_create_regression_pr_full_flow(monkeypatch, tmp_path):
    """Full flow with all external calls monkeypatched."""
    import subprocess
    import urllib.request
    import trial.generator as gen

    # Fake git remote
    class FakeRemoteResult:
        stdout = "https://github.com/acme/my-repo.git\n"

    # Track git calls
    git_calls = []

    def fake_run(cmd, **kwargs):
        git_calls.append(cmd)
        r = FakeRemoteResult()
        r.returncode = 0
        return r

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Fake GitHub API
    pr_responses = []

    class FakeHTTPResponse:
        def __init__(self):
            self._data = json.dumps({"html_url": "https://github.com/acme/my-repo/pull/42"}).encode()

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda req: FakeHTTPResponse())

    # Fake provider
    cfg._default_provider = FakeProvider()

    try:
        pr_url = create_regression_pr(
            USER_MSG,
            RESPONSE,
            github_token="ghp_fake",
            session_id="sess-42",
            repo_path=str(tmp_path),
            tests_dir="tests",
            base_branch="main",
        )
    finally:
        cfg._default_provider = None

    assert pr_url == "https://github.com/acme/my-repo/pull/42"

    # Verify the test file was written
    test_file = tmp_path / "tests" / "test_regression_sess_42.py"
    assert test_file.exists()

    # Verify git was used to create branch, add, commit, push
    cmds = [" ".join(c) for c in git_calls]
    assert any("checkout" in c and "trial/regression" in c for c in cmds)
    assert any("add" in c for c in cmds)
    assert any("commit" in c for c in cmds)
    assert any("push" in c for c in cmds)
