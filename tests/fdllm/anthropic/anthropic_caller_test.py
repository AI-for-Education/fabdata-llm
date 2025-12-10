import pytest
from types import SimpleNamespace

from fdllm import ClaudeCaller
from fdllm.llmtypes import LLMMessage

MESSAGE_ROLES = ("user", "system", "assistant", "error")
TEST_MESSAGE_TEXT = "This is a test"
TEST_MESSAGE = {
    role: LLMMessage(role=role, message=("" if role == "error" else TEST_MESSAGE_TEXT))
    for role in MESSAGE_ROLES
}
TEST_MESSAGE_LIST = [TEST_MESSAGE[role] for role in ("system", "user", "assistant")]
TEST_RESULT_ANTHROPIC = SimpleNamespace(
    content=[SimpleNamespace(text=TEST_MESSAGE_TEXT)]
)


@pytest.mark.parametrize(
    "role, expected",
    [
        (role, {"role": role, "content": message.message})
        for role, message in TEST_MESSAGE.items()
    ],
)
def test_format_message_anthropic(role, expected):
    caller = ClaudeCaller()
    assert caller.format_message(TEST_MESSAGE[role]) == expected


def test_format_messagelist_anthropic():
    caller = ClaudeCaller()
    out = caller.format_messagelist(TEST_MESSAGE_LIST)
    expected = [
        caller.format_message(message)
        for message in TEST_MESSAGE_LIST
        if message.role != "system"
    ]
    assert out == expected


def test_format_output_anthropic():
    caller = ClaudeCaller()
    out = caller.format_output(TEST_RESULT_ANTHROPIC)
    assert isinstance(out, LLMMessage)
