import pytest
from types import SimpleNamespace

from fdllm.llmtypes import LLMMessage, LiteralCaller

TEST_LITERAL_TEXT = "This is a test"
TEST_MESSAGE = LLMMessage(Role="user", Message="Test input")

def test_literal_call():
    caller = LiteralCaller(TEST_LITERAL_TEXT)
    out = caller.call(TEST_MESSAGE)
    assert isinstance(out, LLMMessage)
    assert out.Message == TEST_LITERAL_TEXT
    
async def test_literal_acall(anyio_backend):
    caller = LiteralCaller(TEST_LITERAL_TEXT)
    out = await caller.acall(TEST_MESSAGE)
    assert isinstance(out, LLMMessage)
    assert out.Message == TEST_LITERAL_TEXT
