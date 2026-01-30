"""
End-to-end integration tests for PDF document support.

These tests make real API calls and require valid API keys.
Run with: uv run pytest -m integration
"""
import os
from pathlib import Path

import pytest

# Force load real API keys before any fdllm imports
# (other test files load test.env with override=True during collection)
from dotenv import dotenv_values
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
real_env = dotenv_values(PROJECT_ROOT / ".env")
os.environ.update(real_env)

from fdllm import OpenAICaller, ClaudeCaller, GoogleGenAICaller
from fdllm.llmtypes import LLMMessage, LLMDocument


PROVIDER_CONFIGS = [
    pytest.param(OpenAICaller, "gpt-4o-mini", id="openai"),
    pytest.param(ClaudeCaller, "claude-3-5-haiku-latest", id="anthropic"),
    pytest.param(GoogleGenAICaller, "gemini-test", id="google"),
]


@pytest.mark.integration
@pytest.mark.parametrize("caller_cls,model", PROVIDER_CONFIGS)
def test_pdf_title_extraction(caller_cls, model, sample_pdf_with_title):
    """
    Send a PDF with a known title and ask the model to extract it.
    Verify the title appears in the response.
    """
    # Re-apply real keys (in case other tests overrode them)
    os.environ.update(real_env)

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
@pytest.mark.parametrize("caller_cls,model", PROVIDER_CONFIGS)
async def test_pdf_title_extraction_async(caller_cls, model, sample_pdf_with_title):
    """
    Async version: Send a PDF with a known title and ask the model to extract it.
    """
    # Re-apply real keys (in case other tests overrode them)
    os.environ.update(real_env)

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
