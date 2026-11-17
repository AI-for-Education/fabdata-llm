"""
End-to-end integration tests for PDF document support.

These tests make real API calls and require valid API keys.
Run with: uv run pytest -m integration
"""
import pytest

from fdllm import OpenAICaller, ClaudeCaller, GoogleGenAICaller
from fdllm.llmtypes import LLMMessage, LLMDocument


PROVIDER_CONFIGS = [
    pytest.param(OpenAICaller, "gpt-4.1-mini", "OPENAI_API_KEY", id="openai"),
    pytest.param(
        ClaudeCaller,
        "claude-test",
        "ANTHROPIC_API_KEY",
        id="anthropic",
    ),
    pytest.param(GoogleGenAICaller, "gemini-test", "GEMINI_API_KEY", id="google"),
]


@pytest.mark.integration
@pytest.mark.parametrize("caller_cls,model,env_var", PROVIDER_CONFIGS)
def test_pdf_title_extraction(
    caller_cls, model, env_var, sample_pdf_with_title, require_api_key
):
    """
    Send a PDF with a known title and ask the model to extract it.
    Verify the title appears in the response.
    """
    require_api_key(env_var)

    pdf_bytes, expected_title = sample_pdf_with_title

    caller = caller_cls(model=model)
    doc = LLMDocument(Data=pdf_bytes, Filename="test_document.pdf")

    message = LLMMessage(
        Role="user",
        Message="What is the title of this document? Reply with just the title.",
        Documents=[doc],
    )

    response = caller.call(message, max_tokens=256)

    assert response.Message is not None, "Response message should not be None"
    assert expected_title in response.Message, (
        f"Expected title '{expected_title}' not found in response: {response.Message}"
    )


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.parametrize("caller_cls,model,env_var", PROVIDER_CONFIGS)
async def test_pdf_title_extraction_async(
    caller_cls, model, env_var, sample_pdf_with_title, require_api_key
):
    """
    Async version: Send a PDF with a known title and ask the model to extract it.
    """
    require_api_key(env_var)

    pdf_bytes, expected_title = sample_pdf_with_title

    caller = caller_cls(model=model)
    doc = LLMDocument(Data=pdf_bytes, Filename="test_document.pdf")

    message = LLMMessage(
        Role="user",
        Message="What is the title of this document? Reply with just the title.",
        Documents=[doc],
    )

    response = await caller.acall(message, max_tokens=256)

    assert response.Message is not None, "Response message should not be None"
    assert expected_title in response.Message, (
        f"Expected title '{expected_title}' not found in response: {response.Message}"
    )
