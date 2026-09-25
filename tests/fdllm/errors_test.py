"""
Tests for typed response errors and the response-level retry in LLMCaller.call/acall.
"""
import asyncio
from types import SimpleNamespace

import pytest
from tenacity import wait_none

from google.genai.types import FinishReason

from fdllm import OpenAICaller
from fdllm.constants import LLM_DEFAULT_MAX_RESPONSE_RETRIES
from fdllm.errors import (
    EmptyLLMResponse,
    InvalidProviderResponse,
    LLMContentFiltered,
    LLMResponseError,
    empty_response_error,
    is_retryable_response_error,
    safe_usage,
)
from fdllm.llmtypes import LLMMessage

TEST_MODEL = "gpt-4.1-mini"


# ===== Error types =====

def test_errors_are_value_errors():
    """Existing `except ValueError` handlers keep catching response errors."""
    for cls in (LLMResponseError, InvalidProviderResponse, EmptyLLMResponse, LLMContentFiltered):
        assert issubclass(cls, ValueError)
    assert issubclass(LLMContentFiltered, EmptyLLMResponse)


@pytest.mark.parametrize(
    "error, retryable",
    [
        (InvalidProviderResponse("x"), False),
        (EmptyLLMResponse("x"), True),
        (EmptyLLMResponse("x", stop_reason="end_turn"), True),
        (EmptyLLMResponse("x", stop_reason="max_tokens"), False),
        (EmptyLLMResponse("x", stop_reason="length"), False),
        (EmptyLLMResponse("x", stop_reason=FinishReason.MAX_TOKENS), False),
        (LLMContentFiltered("x", stop_reason="end_turn"), False),
        (EmptyLLMResponse("x", stop_reason="end_turn", retryable=False), False),
    ],
)
def test_default_retryable(error, retryable):
    assert error.retryable is retryable
    assert is_retryable_response_error(error) is retryable


def test_is_retryable_ignores_other_exceptions():
    assert is_retryable_response_error(ValueError("x")) is False


@pytest.mark.parametrize(
    "stop_reason, error_cls",
    [
        ("end_turn", EmptyLLMResponse),
        (None, EmptyLLMResponse),
        ("refusal", LLMContentFiltered),
        ("content_filter", LLMContentFiltered),
        (FinishReason.SAFETY, LLMContentFiltered),
        ("guardrail_intervened", LLMContentFiltered),
    ],
)
def test_empty_response_error_picks_class(stop_reason, error_cls):
    assert type(empty_response_error("x", stop_reason=stop_reason)) is error_cls


def test_str_includes_safe_metadata():
    err = EmptyLLMResponse(
        "Empty response: no choices",
        provider="openai",
        model="m",
        stop_reason=FinishReason.STOP,
        block_types=["thinking"],
        usage={"total_tokens": 3},
    )
    assert str(err) == (
        "Empty response: no choices (provider='openai', model='m', "
        "stop_reason='STOP', block_types=['thinking'], "
        "usage={'total_tokens': 3}, retryable=True)"
    )


def test_safe_usage_keeps_only_int_counts():
    usage = SimpleNamespace(input_tokens=3, output_tokens=None, extra="text")
    assert safe_usage(usage, "input_tokens", "output_tokens", "extra") == {"input_tokens": 3}
    assert safe_usage({"a": 1, "b": "x"}, "a", "b") == {"a": 1}
    assert safe_usage(None, "a") is None
    assert safe_usage(SimpleNamespace(), "a") is None


# ===== Response-level retry in call / acall =====

def _ok_response():
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="hello", tool_calls=None),
                logprobs=None,
                finish_reason="stop",
            )
        ],
        usage=None,
    )


def _empty_response(finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=None),
                logprobs=None,
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


@pytest.fixture
def no_wait(monkeypatch):
    monkeypatch.setattr("fdllm.llmtypes.wait_exponential", lambda **kwargs: wait_none())


def _caller_returning(responses):
    """OpenAICaller whose API function returns the given responses in order."""
    caller = OpenAICaller(TEST_MODEL)
    calls = []

    def func(**kwargs):
        calls.append(kwargs)
        return responses[len(calls) - 1]

    async def afunc(**kwargs):
        return func(**kwargs)

    caller.Func = func
    caller.AFunc = afunc
    return caller, calls


def test_call_retries_retryable_empty_response(no_wait):
    caller, calls = _caller_returning([_empty_response(), _ok_response()])

    out = caller.call(LLMMessage(Role="user", Message="hi"))

    assert out.Message == "hello"
    assert len(calls) == 2


def test_acall_retries_retryable_empty_response(no_wait):
    caller, calls = _caller_returning([_empty_response(), _ok_response()])

    out = asyncio.run(caller.acall(LLMMessage(Role="user", Message="hi")))

    assert out.Message == "hello"
    assert len(calls) == 2


def test_call_gives_up_after_max_response_retries(no_wait):
    responses = [_empty_response()] * LLM_DEFAULT_MAX_RESPONSE_RETRIES
    caller, calls = _caller_returning(responses)

    with pytest.raises(EmptyLLMResponse):
        caller.call(LLMMessage(Role="user", Message="hi"))
    assert len(calls) == LLM_DEFAULT_MAX_RESPONSE_RETRIES


@pytest.mark.parametrize(
    "finish_reason, error_cls",
    [("content_filter", LLMContentFiltered), ("length", EmptyLLMResponse)],
)
def test_call_does_not_retry_non_retryable_response(no_wait, finish_reason, error_cls):
    caller, calls = _caller_returning([_empty_response(finish_reason), _ok_response()])

    with pytest.raises(error_cls):
        caller.call(LLMMessage(Role="user", Message="hi"))
    assert len(calls) == 1
