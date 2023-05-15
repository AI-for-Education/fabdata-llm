import pytest

import anthropic
from anthropic.tokenizer import get_tokenizer

from fdllm import ClaudeCaller
from fdllm.llmtypes import LLMMessage

MESSAGE_ROLES = ("user", "system", "assistant", "error")
TEST_MESSAGE_TEXT = "This is a test"
TEST_MESSAGE = {
    role: LLMMessage(Role=role, Message=("" if role == "error" else TEST_MESSAGE_TEXT))
    for role in MESSAGE_ROLES
}
TEST_MESSAGE_LIST = [TEST_MESSAGE[role] for role in ("system", "user", "assistant")]
TEST_RESULT_ANTHROPIC = {"completion": f" {TEST_MESSAGE_TEXT}"}


@pytest.mark.parametrize(
    "role, expected",
    [
        (
            role,
            f"{anthropic.HUMAN_PROMPT} {message.Message}{anthropic.AI_PROMPT}"
            if role in ["user", "system"]
            else f"{anthropic.AI_PROMPT} {message.Message}"
            if role in ["assistant"]
            else "",
        )
        for role, message in TEST_MESSAGE.items()
    ],
)
def test_format_message_anthropic(role, expected):
    caller = ClaudeCaller()
    assert caller.format_message(TEST_MESSAGE[role]) == expected


def test_format_messagelist_anthropic():
    caller = ClaudeCaller()
    out = caller.format_messagelist(TEST_MESSAGE_LIST)
    expected = "".join(caller.format_message(message) for message in TEST_MESSAGE_LIST)
    assert out == expected


def test_format_output_anthropic():
    caller = ClaudeCaller()
    out = caller.format_output(TEST_RESULT_ANTHROPIC)
    assert isinstance(out, LLMMessage)


def test_tokenize_anthropic():
    caller = ClaudeCaller()
    out = len(caller.tokenize(TEST_MESSAGE_LIST))
    expected = len(get_tokenizer().encode(caller.format_messagelist(TEST_MESSAGE_LIST)))
    assert out == expected
