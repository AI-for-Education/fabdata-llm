"""
End-to-end version of the non-retryable empty response tests in errors_test.py.

Reasoning/thinking is enabled with a max_tokens too small for the model to
finish thinking, so no visible output is produced. The caller must raise
EmptyLLMResponse and must not retry the request.

These tests make real API calls and require valid API keys.
Run with: uv run pytest -m integration tests/fdllm/integration/test_empty_response_e2e.py
"""
import pytest
from tenacity import wait_none

from fdllm import OpenAICaller, ClaudeCaller, GoogleGenAICaller
from fdllm.errors import EmptyLLMResponse
from fdllm.llmtypes import LLMMessage

PROMPT = "List every prime number below 500, then compute the sum of their squares."

PROVIDER_CONFIGS = [
    pytest.param(
        OpenAICaller,
        "gpt-test",
        "OPENAI_API_KEY",
        dict(max_tokens=16, reasoning_effort="low"),
        id="openai",
    ),
    # Anthropic requires budget_tokens >= 1024 and max_tokens > budget_tokens
    pytest.param(
        ClaudeCaller,
        "claude-test",
        "ANTHROPIC_API_KEY",
        dict(max_tokens=16, output_config={"effort": "medium"}),
        id="anthropic",
    ),
    # "low" can skip thinking entirely and return truncated text instead
    pytest.param(
        GoogleGenAICaller,
        "gemini-test",
        "GEMINI_API_KEY",
        dict(max_tokens=16, thinking_config={"thinking_level": "medium"}),
        id="google",
    ),
]


@pytest.fixture
def no_wait(monkeypatch):
    monkeypatch.setattr("fdllm.llmtypes.wait_exponential", lambda **kwargs: wait_none())


def _counting_caller(caller_cls, model):
    """Caller whose real API function records each request sent."""
    caller = caller_cls(model=model)
    calls = []
    func = caller.Func

    def counting_func(**kwargs):
        calls.append(kwargs)
        return func(**kwargs)

    caller.Func = counting_func
    return caller, calls


@pytest.mark.integration
@pytest.mark.parametrize("caller_cls,model,env_var,call_kwargs", PROVIDER_CONFIGS)
def test_call_does_not_retry_token_exhaustion(
    caller_cls, model, env_var, call_kwargs, require_api_key, no_wait
):
    require_api_key(env_var)
    caller, calls = _counting_caller(caller_cls, model)

    with pytest.raises(EmptyLLMResponse) as excinfo:
        caller.call(LLMMessage(Role="user", Message=PROMPT), **call_kwargs)

    assert excinfo.value.retryable is False, str(excinfo.value)
    assert len(calls) == 1
