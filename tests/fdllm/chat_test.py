import pytest
from unittest.mock import patch
from types import SimpleNamespace
from itertools import product
from typing import List
from pathlib import Path

from pydantic import PrivateAttr

from fdllm.chat import ChatController, ChatPlugin
from fdllm import OpenAICaller, ClaudeCaller, GoogleGenAICaller
from fdllm.llmtypes import LLMMessage
from fdllm.openai.tokenizer import tokenize_chatgpt_messages
from fdllm.sysutils import register_models
from fdllm.constants import LLM_DEFAULT_MAX_TOKENS

register_models(Path.home() / ".fdllm/custom_models.yaml")

TEST_PLUGIN_SYSMSG = {0: "A", -1: "B", -2: "C"}

class TESTPLUGIN(ChatPlugin):
    Restore_Attrs: List[str] = ["Sys_Msg"]
    _history: List[LLMMessage] = PrivateAttr()

    def pre_chat(self, prompt: str, *args, **kwargs):
        self.Controller.Sys_Msg = TEST_PLUGIN_SYSMSG

    async def pre_achat(self, prompt: str, *args, **kwargs):
        self.Controller.Sys_Msg = TEST_PLUGIN_SYSMSG
    
    def post_chat(self, result: LLMMessage, *args, **kwargs):
        self._history = self.Controller.History.copy()
        return result
    
    async def post_achat(self, result: LLMMessage, *args, **kwargs):
        self._history = self.Controller.History.copy()
        return result

    def register(self):
        return super().register()

    def unregister(self):
        return super().unregister()


TEST_PROMPT_TEXT = "This is a user test"
TEST_SYSTEM_TEXT = "This is a system test"
TEST_OUTPUT_TEXT = "This is an output test"
TEST_PROMPT_MESSAGE = LLMMessage(role="user", message=TEST_PROMPT_TEXT)
TEST_SYSTEM_MESSAGE = LLMMessage(role="system", message=TEST_SYSTEM_TEXT)
TEST_LLM_OUTPUT = LLMMessage(role="assistant", message=TEST_OUTPUT_TEXT)
TEST_SYSMSG_FORMAT = {
    (0,): [TEST_SYSTEM_MESSAGE, TEST_PROMPT_MESSAGE],
    (-1,): [TEST_PROMPT_MESSAGE, TEST_SYSTEM_MESSAGE],
    (-2,): [TEST_SYSTEM_MESSAGE, TEST_PROMPT_MESSAGE],
    (0, -1): [
        TEST_SYSTEM_MESSAGE,
        TEST_PROMPT_MESSAGE,
        TEST_SYSTEM_MESSAGE,
    ],
    (0, -2): [
        TEST_SYSTEM_MESSAGE,
        TEST_SYSTEM_MESSAGE,
        TEST_PROMPT_MESSAGE,
    ],
    (-1, -2): [
        TEST_SYSTEM_MESSAGE,
        TEST_PROMPT_MESSAGE,
        TEST_SYSTEM_MESSAGE,
    ],
    (0, -1, -2): [
        TEST_SYSTEM_MESSAGE,
        TEST_SYSTEM_MESSAGE,
        TEST_PROMPT_MESSAGE,
        TEST_SYSTEM_MESSAGE,
    ],
}
TEST_CALLERS = [OpenAICaller, ClaudeCaller, GoogleGenAICaller]


@pytest.mark.parametrize(
    "caller, retval",
    [(caller, TEST_LLM_OUTPUT) for caller in TEST_CALLERS],
)
def test_chat(caller, retval):
    controller = ChatController(Caller=caller())
    with patch(f"{caller.__module__}.{caller.__name__}.call", return_value=retval):
        new_message, result = controller.chat(TEST_PROMPT_TEXT)
    assert isinstance(new_message, LLMMessage)
    assert result == retval


@pytest.mark.parametrize(
    "caller, retval", [(caller, TEST_LLM_OUTPUT) for caller in TEST_CALLERS]
)
async def test_achat(anyio_backend, caller, retval):
    controller = ChatController(Caller=caller())
    with patch(f"{caller.__module__}.{caller.__name__}.acall", return_value=retval):
        new_message, result = await controller.achat(TEST_PROMPT_TEXT)
    assert isinstance(new_message, LLMMessage)
    assert result == retval


@pytest.mark.parametrize(
    "sys_msg_indices, expected",
    [(smi, thf) for smi, thf in TEST_SYSMSG_FORMAT.items()],
)
def test_prechat(sys_msg_indices, expected):
    sys_msg = {idx: TEST_SYSTEM_TEXT for idx in sys_msg_indices}
    caller = TEST_CALLERS[0]
    controller = ChatController(Caller=caller(), Sys_Msg=sys_msg)
    new_message, latest_convo = controller._prechat(
        TEST_PROMPT_TEXT, LLM_DEFAULT_MAX_TOKENS
    )
    assert isinstance(new_message, LLMMessage)
    assert latest_convo == expected


def test_plugin():
    sys_msg = {idx: TEST_SYSTEM_TEXT for idx in [0, -1]}
    caller = TEST_CALLERS[0]
    plugin = TESTPLUGIN(Caller=caller())
    controller = ChatController(Caller=caller(), Sys_Msg=sys_msg)
    controller.register_plugin(plugin)
    with patch(
        f"{caller.__module__}.{caller.__name__}.call", return_value=TEST_LLM_OUTPUT
    ):
        new_message, result = controller.chat(TEST_PROMPT_TEXT)
    assert result == TEST_LLM_OUTPUT
    assert controller.Sys_Msg == sys_msg
    assert plugin._history == controller.History
    controller._run_plugins(TEST_PROMPT_TEXT)
    assert controller.Sys_Msg == TEST_PLUGIN_SYSMSG
    controller._clean_plugins(result)
    assert controller.Sys_Msg == sys_msg
    controller.unregister_plugin(plugin)
    assert controller._plugins == []
    
async def test_aplugin(anyio_backend):
    sys_msg = {idx: TEST_SYSTEM_TEXT for idx in [0, -1]}
    caller = TEST_CALLERS[0]
    plugin = TESTPLUGIN(Caller=caller())
    controller = ChatController(Caller=caller(), Sys_Msg=sys_msg)
    controller.register_plugin(plugin)
    with patch(
        f"{caller.__module__}.{caller.__name__}.acall", return_value=TEST_LLM_OUTPUT
    ):
        new_message, result = await controller.achat(TEST_PROMPT_TEXT)
    assert result == TEST_LLM_OUTPUT
    assert controller.Sys_Msg == sys_msg
    assert plugin._history == controller.History
    await controller._arun_plugins(TEST_PROMPT_TEXT)
    assert controller.Sys_Msg == TEST_PLUGIN_SYSMSG
    await controller._aclean_plugins(result)
    assert controller.Sys_Msg == sys_msg
    controller.unregister_plugin(plugin)
    assert controller._plugins == []
