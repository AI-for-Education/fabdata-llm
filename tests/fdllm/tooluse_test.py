import pytest
from unittest.mock import patch
from unittest import mock
from types import SimpleNamespace
from itertools import product
from typing import List

from pydantic import PrivateAttr, Field

from fdllm import get_caller
from fdllm.chat import ChatController
from fdllm.llmtypes import LLMMessage, LLMToolCall
from fdllm.tooluse import (
    Tool,
    ToolParam,
    ToolUsePlugin,
    ToolInvalidParamError,
    ToolMissingParamError,
)


class TESTTOOL1(Tool):
    name = "mul"
    description = "Multiply 2 numbers"
    params = {
        "x": ToolParam(type="number", required=True),
        "y": ToolParam(type="number", required=True),
    }

    def execute(self, **params):
        res = params["x"] * params["y"]
        return f"{res :.4f}"

    async def aexecute(self, **params):
        return self.execute()


class TESTTOOL2(Tool):
    name = "add"
    description = "Add 2 numbers"
    params = {
        "x": ToolParam(type="number", required=True),
        "y": ToolParam(type="number", required=True),
    }

    def execute(self, **params):
        res = params["x"] + params["y"]
        return f"{res :.4f}"

    async def aexecute(self, **params):
        return self.execute()


class TESTTOOLPLUGIN(ToolUsePlugin):
    Tools: List[Tool] = Field(default_factory=lambda: [TESTTOOL1(), TESTTOOL2()])


TEST_PROMPT_TEXT = "This is a user test"
TEST_OUTPUT_TEXT = "This is an output test"
TEST_PROMPT_MESSAGE = LLMMessage(role="user", message=TEST_PROMPT_TEXT)
TEST_LLM_OUTPUT = LLMMessage(role="assistant", message=TEST_OUTPUT_TEXT)

TEST_TOOL_CALLS = {
    "valid_single": [LLMToolCall(id="testtc1", name="mul", args={"x": 2, "y": 4})],
    "missing_single": [LLMToolCall(id="testtc1", name="mul", args={"x": 2})],
    "extra_single": [
        LLMToolCall(id="testtc1", name="mul", args={"x": 2, "y": 4, "z": 6})
    ],
    "valid_multi": [
        LLMToolCall(id="testtc1", name="mul", args={"x": 2, "y": 4}),
        LLMToolCall(id="testtc2", name="add", args={"x": 6, "y": 8}),
    ],
    "missing_multi": [
        LLMToolCall(id="testtc1", name="mul", args={"x": 2}),
        LLMToolCall(id="testtc2", name="add", args={"x": 6}),
    ],
    "extra_multi": [
        LLMToolCall(id="testtc1", name="mul", args={"x": 2, "y": 4, "z": 6}),
        LLMToolCall(id="testtc2", name="add", args={"x": 6, "y": 8, "z": 10}),
    ],
}
TEST_TOOL_CALL_OUTPUTS = {
    key: LLMMessage(role="assistant", tool_calls=tc)
    for key, tc in TEST_TOOL_CALLS.items()
}

TEST_EXCEPTION = {
    "missing_single": ToolMissingParamError,
    "missing_multi": ToolMissingParamError,
    "extra_single": ToolInvalidParamError,
    "extra_multi": ToolInvalidParamError,
}

TEST_CALLERS = ["gpt-3.5-turbo-1106", "claude-3-haiku-20240307"]


@pytest.mark.parametrize(
    "toolstr, caller, test_over",
    [
        (tc, cl, to)
        for tc, cl, to in product(TEST_TOOL_CALLS, TEST_CALLERS, (True, False))
    ],
)
def test_toolcall(toolstr: str, caller: str, test_over):
    validstr = f"valid_{toolstr.split('_')[-1]}"
    caller = get_caller(caller)
    plugin = TESTTOOLPLUGIN()
    if toolstr.startswith(("missing", "extra")):
        nreps = plugin._max_tool_attempt + int(test_over)
    else:
        nreps = 0
    controller = ChatController(Caller=caller)
    controller.register_plugin(plugin)
    with patch(
        f"{caller.__class__.__module__}.{caller.__class__.__name__}.call",
        side_effect=[
            *[TEST_TOOL_CALL_OUTPUTS[toolstr]] * nreps,
            TEST_TOOL_CALL_OUTPUTS[validstr],
            TEST_LLM_OUTPUT,
        ],
    ):
        try:
            new_message, result = controller.chat(TEST_PROMPT_TEXT)
        except Exception as e:
            assert isinstance(e, TEST_EXCEPTION[toolstr])
            assert len(controller.History) == 0
            return
    assert result == TEST_LLM_OUTPUT
    assert new_message == TEST_PROMPT_MESSAGE
    assert controller.History[0] == TEST_PROMPT_MESSAGE
    assert controller.History[1] == TEST_TOOL_CALL_OUTPUTS[validstr]
    assert controller.History[3] == TEST_LLM_OUTPUT
    assert len(controller.History) == 4
    toolresult = controller.History[2]
    toolresultmanual = [
        pt.execute(**tc.args)
        for pt, tc in zip(plugin.Tools, TEST_TOOL_CALL_OUTPUTS[validstr].tool_calls)
    ]
    assert [tc.response for tc in toolresult.tool_calls] == toolresultmanual
