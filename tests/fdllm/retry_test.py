"""
Tests for the retry decision (fdllm.retry.should_retry) and Retry-After handling.

Errors are built with each SDK's own constructors from real HTTP responses, so
the tests exercise the exception classes the SDKs actually raise.
"""
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import aiohttp
import anthropic
import httpx
import openai
import pytest
from google.genai import errors as genai_errors

from fdllm.errors import (
    EmptyLLMResponse,
    LLMContentFiltered,
    LLMProviderError,
    LLMProviderRejected,
    is_retryable_status,
    provider_error,
)
from fdllm.retry import retry_after_seconds, should_retry, wait_retry_after

_REQUEST = httpx.Request("POST", "https://example.invalid/v1/x")
_OPENAI = openai.OpenAI(api_key="x")
_ANTHROPIC = anthropic.Anthropic(api_key="x")

RETRYABLE = [408, 409, 429, 500, 502, 503, 504, 529]
NOT_RETRYABLE = [400, 401, 402, 403, 404, 413, 422]


def _response(status, body=None, headers=None):
    return httpx.Response(status, request=_REQUEST, json=body or {}, headers=headers)


def openai_error(status, code=None, headers=None):
    body = {"error": {"message": "m", "type": "t", "code": code}}
    return _OPENAI._make_status_error_from_response(_response(status, body, headers))


def anthropic_error(status, headers=None):
    body = {"type": "error", "error": {"type": "t", "message": "m"}}
    return _ANTHROPIC._make_status_error_from_response(_response(status, body, headers))


def google_error(status):
    body = {"error": {"code": status, "message": "m", "status": "S"}}
    try:
        genai_errors.APIError.raise_for_response(_response(status, body))
    except genai_errors.APIError as e:
        return e
    raise AssertionError("google-genai did not raise")


ERROR_BUILDERS = [openai_error, anthropic_error, google_error]


# ===== Status codes =====


@pytest.mark.parametrize("build", ERROR_BUILDERS, ids=["openai", "anthropic", "google"])
@pytest.mark.parametrize("status", RETRYABLE)
def test_transient_statuses_are_retried(build, status):
    assert should_retry(build(status))


@pytest.mark.parametrize("build", ERROR_BUILDERS, ids=["openai", "anthropic", "google"])
@pytest.mark.parametrize("status", NOT_RETRYABLE)
def test_permanent_statuses_are_not_retried(build, status):
    assert not should_retry(build(status))


def test_insufficient_quota_is_not_retried():
    assert should_retry(openai_error(429, code="rate_limit_exceeded"))
    assert not should_retry(openai_error(429, code="insufficient_quota"))


@pytest.mark.parametrize("build", [openai_error, anthropic_error], ids=["openai", "anthropic"])
def test_x_should_retry_header_wins(build):
    assert should_retry(build(400, headers={"x-should-retry": "true"}))
    assert not should_retry(build(503, headers={"x-should-retry": "false"}))


@pytest.mark.parametrize(
    "error_type, retryable",
    [("overloaded_error", True), ("api_error", True), ("invalid_request_error", False)],
)
def test_anthropic_error_event_inside_stream(error_type, retryable):
    # the SDK builds these from the open (HTTP 200) stream response
    exc = _ANTHROPIC._make_status_error(
        "m",
        body={"type": "error", "error": {"type": error_type, "message": "m"}},
        response=_response(200),
    )
    assert should_retry(exc) is retryable


# ===== Connection errors and other exceptions =====


@pytest.mark.parametrize(
    "exc",
    [
        openai.APIConnectionError(request=_REQUEST),
        openai.APITimeoutError(request=_REQUEST),
        anthropic.APIConnectionError(request=_REQUEST),
        anthropic.APITimeoutError(request=_REQUEST),
        httpx.ReadError("Connection reset by peer"),
        httpx.ConnectError("refused"),
        httpx.ReadTimeout("timed out"),
        httpx.RemoteProtocolError("peer closed connection"),
        aiohttp.ServerDisconnectedError(),
    ],
    ids=lambda e: f"{type(e).__module__.split('.')[0]}.{type(e).__name__}",
)
def test_connection_errors_are_retried(exc):
    assert should_retry(exc)


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ProxyError("bad proxy"),
        httpx.UnsupportedProtocol("ftp://"),
        ValueError("bug"),
        KeyError("bug"),
        RuntimeError("bug"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_configuration_errors_and_bugs_are_not_retried(exc):
    assert not should_retry(exc)


def test_response_errors_decide_for_themselves():
    assert should_retry(EmptyLLMResponse("empty", stop_reason="stop"))
    assert not should_retry(EmptyLLMResponse("empty", stop_reason="length"))
    assert not should_retry(LLMContentFiltered("filtered"))


# ===== Provider-reported errors inside a 200 response =====


@pytest.mark.parametrize("code", RETRYABLE)
def test_provider_error_with_transient_code(code):
    err = provider_error("x", code=code)
    assert type(err) is LLMProviderError
    assert should_retry(err)
    assert err.details == {"code": str(code)}


@pytest.mark.parametrize("code", NOT_RETRYABLE)
def test_provider_error_with_permanent_code(code):
    err = provider_error("x", code=code)
    assert isinstance(err, LLMProviderRejected)
    assert not should_retry(err)


@pytest.mark.parametrize("code", [None, "502", "rate_limit", True, ["x"]])
def test_provider_error_without_usable_code_is_retried(code):
    err = provider_error("x", code=code)
    assert type(err) is LLMProviderError
    assert should_retry(err)


def test_is_retryable_status():
    assert all(is_retryable_status(c) for c in RETRYABLE + [501, 599])
    assert not any(is_retryable_status(c) for c in NOT_RETRYABLE)


# ===== Retry-After =====


@pytest.mark.parametrize(
    "headers, expected",
    [
        ({"retry-after": "3"}, 3.0),
        ({"retry-after": "1.5"}, 1.5),
        ({"retry-after-ms": "250"}, 0.25),
        ({"retry-after-ms": "250", "retry-after": "9"}, 0.25),
        ({"retry-after": "0"}, None),
        ({"retry-after": "61"}, None),
        ({"retry-after": "soon"}, None),
        ({}, None),
    ],
)
def test_retry_after_seconds(headers, expected):
    assert retry_after_seconds(openai_error(429, headers=headers)) == expected


def test_retry_after_http_date():
    when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30), usegmt=True)
    seconds = retry_after_seconds(anthropic_error(429, headers={"retry-after": when}))
    assert 25 < seconds <= 30


def test_retry_after_without_response():
    assert retry_after_seconds(httpx.ReadError("reset")) is None


def _state(exc):
    return SimpleNamespace(outcome=SimpleNamespace(exception=lambda: exc))


def test_wait_uses_retry_after_else_fallback():
    wait = wait_retry_after(lambda retry_state: 7.0)
    assert wait(_state(openai_error(429, headers={"retry-after": "2"}))) == 2.0
    assert wait(_state(openai_error(429))) == 7.0
    assert wait(_state(httpx.ReadError("reset"))) == 7.0
