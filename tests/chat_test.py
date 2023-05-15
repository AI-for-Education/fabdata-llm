import pytest
from unittest.mock import patch
from types import SimpleNamespace

import anthropic
from anthropic.tokenizer import get_tokenizer

from fdllm.chat import ChatController
from fdllm import GPTCaller, ClaudeCaller
from fdllm.llmtypes import LLMMessage
from fdllm.openai.tokenizer import tokenize_chatgpt_messages

TEST_MESSAGE_TEXT = "This is a test"
TEST_LLM_OUTPUT = LLMMessage(Role="assistant", Message=TEST_MESSAGE_TEXT)
TEST_CALLERS = [GPTCaller, ClaudeCaller]

@pytest.mark.parametrize(
    "caller, retval", [(caller, TEST_LLM_OUTPUT) for caller in TEST_CALLERS]
)
def test_chat(caller, retval):
    controller = ChatController(Caller=caller())
    with patch(
        f"{caller.__module__}.{caller.__name__}.call",
        return_value=retval
    ):
        new_message, result = controller.chat(TEST_MESSAGE_TEXT)
        assert isinstance(new_message, LLMMessage)
        assert result == retval

@pytest.mark.parametrize(
    "caller, retval", [(caller, TEST_LLM_OUTPUT) for caller in TEST_CALLERS]
)
async def test_achat(anyio_backend, caller, retval):
    controller = ChatController(Caller=caller())
    with patch(
        f"{caller.__module__}.{caller.__name__}.acall",
        return_value=retval
    ):
        new_message, result = await controller.achat(TEST_MESSAGE_TEXT)
        assert isinstance(new_message, LLMMessage)
        assert result == retval