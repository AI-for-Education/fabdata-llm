import pytest
from types import SimpleNamespace
from pathlib import Path

from fdllm import GPTCaller, GPTVisionCaller
from fdllm.llmtypes import LLMMessage
from fdllm.openai.tokenizer import tokenize_chatgpt_messages
from fdllm.sysutils import register_models

MESSAGE_ROLES = ("user", "system", "assistant", "error")
TEST_MESSAGE_TEXT = "This is a test"
TEST_MESSAGE = {
    role: LLMMessage(Role=role, Message=("" if role == "error" else TEST_MESSAGE_TEXT))
    for role in MESSAGE_ROLES
}
TEST_MESSAGE_LIST = [TEST_MESSAGE[role] for role in ("system", "user", "assistant")]
TEST_RESULT_OPENAI = SimpleNamespace(
    choices=[SimpleNamespace(
        message=SimpleNamespace(content=TEST_MESSAGE_TEXT)
    )]
)
TEST_MODELS = ["gpt-3.5-turbo", "fabdata-openai-eastus2-gpt35"]
TEST_VISION_MODELS = ["gpt-4-vision-preview"]

register_models(Path.home() / ".fdllm/custom_models.yaml")

@pytest.mark.parametrize("model", TEST_MODELS)
def test_init_openai(model):
    GPTCaller(model=model)

@pytest.mark.parametrize("model", TEST_VISION_MODELS)
def test_init_openaivision(model):
    GPTVisionCaller(model=model)

@pytest.mark.parametrize(
    "role, expected",
    [
        (role, {"role": role, "content": message.Message})
        for role, message in TEST_MESSAGE.items()
    ],
)
def test_format_message_openai(role, expected):
    caller = GPTCaller()
    assert caller.format_message(TEST_MESSAGE[role]) == expected


def test_format_messagelist_openai():
    caller = GPTCaller()
    out = caller.format_messagelist(TEST_MESSAGE_LIST)
    expected = [caller.format_message(message) for message in TEST_MESSAGE_LIST]
    assert out == expected


def test_format_output_openai():
    caller = GPTCaller()
    out = caller.format_output(TEST_RESULT_OPENAI)
    assert isinstance(out, LLMMessage)


def test_tokenize_openai():
    caller = GPTCaller()
    out = len(caller.tokenize(TEST_MESSAGE_LIST))
    expected = len(
        tokenize_chatgpt_messages(caller.format_messagelist(TEST_MESSAGE_LIST))[0]
    )
    assert out == expected
