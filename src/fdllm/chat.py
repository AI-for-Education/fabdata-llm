from __future__ import annotations
from typing import List, Literal, Optional, Any, Dict, ClassVar, Generator
from abc import ABC, abstractmethod
import copy

from PIL import Image
from pydantic import BaseModel, Field, PrivateAttr

from .constants import (
    LLM_EMPTY_QUESTION,
    LLM_QUESTION_TOO_LONG,
    LLM_INAPPROPRIATE_QUESTION,
    LLM_DEFAULT_MAX_TOKENS,
)
from .llmtypes import LLMMessage, LLMCaller, LLMImage


class ChatController(BaseModel):
    Caller: LLMCaller
    History: List[LLMMessage] = Field(default_factory=list)
    Sys_Msg: Dict[Literal[0, -1, -2], str] = Field(default_factory=dict)
    Sys_Msg_Confirmation: Dict[Literal[0, -1, -2], str] = Field(default_factory=dict)
    Keep_History: bool = True
    _plugins: List[ChatPlugin] = PrivateAttr(default_factory=list)

    @property
    def recent_history(self):
        return next(iter(self.interactions(reverse=True)))

    @property
    def recent_tool_calls(self):
        return next(iter(self.tool_calls(reverse=True)))
    
    @property
    def recent_tool_responses(self):
        return next(iter(self.tool_responses(reverse=True)))

    def interactions(self, reverse=False) -> Generator[List[LLMMessage]]:
        histbuff = []
        if reverse:
            for msg in self.History[-1::-1]:
                histbuff.append(msg)
                if msg.Role == "user":
                    yield histbuff[-1::-1]
                    histbuff = []
        else:
            for msg in self.History:
                histbuff.append(msg)
                if msg.Role == "assistant" and msg.ToolCalls is None:
                    yield histbuff
                    histbuff = []

    def tool_calls(self, reverse=False):
        for interaction in self.interactions(reverse):
            yield [
                msg
                for msg in interaction
                if msg.ToolCalls is not None and msg.ToolCalls[0].Response is None
            ]
    
    def tool_responses(self, reverse=False):
        for interaction in self.interactions(reverse):
            yield [
                msg
                for msg in interaction
                if msg.ToolCalls is not None and msg.ToolCalls[0].Response is not None
            ]

    def chat(
        self,
        prompt: str = "",
        max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
        response_schema: Optional[BaseModel] = None,
        images: Optional[List[Image.Image]] = None,
        detail: Literal["low", "high"] = "low",
        **kwargs,
    ):
        self._run_plugins(prompt, max_tokens, response_schema, images, detail, **kwargs)
        try:
            new_message, latest_convo = self._prechat(
                prompt, max_tokens, images, detail
            )
        except:
            self._clean_plugins(None, max_tokens, response_schema, images, detail, **kwargs)
            raise
        if new_message is not None and new_message.Role == "error":
            self._clean_plugins(None, max_tokens, response_schema, images, detail, **kwargs)
            return new_message, None
        result = self.Caller.call(latest_convo, max_tokens, response_schema, **kwargs)
        self._postchat(result)
        result = self._clean_plugins(result, max_tokens, response_schema, images, detail, **kwargs)
        return new_message, result

    async def achat(
        self,
        prompt: str = "",
        max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
        response_schema: Optional[BaseModel] = None,
        images: Optional[List[Image.Image]] = None,
        detail: Literal["low", "high"] = "low",
        **kwargs,
    ):
        await self._arun_plugins(prompt, max_tokens, response_schema, images, detail, **kwargs)
        try:
            new_message, latest_convo = self._prechat(
                prompt, max_tokens, images, detail
            )
        except:
            await self._aclean_plugins(None, max_tokens, response_schema, images, detail, **kwargs)
            raise
        if new_message is not None and new_message.Role == "error":
            await self._aclean_plugins(None, max_tokens, response_schema, images, detail, **kwargs)
            return new_message, None
        result = await self.Caller.acall(latest_convo, max_tokens, response_schema, **kwargs)
        self._postchat(result)
        result = await self._aclean_plugins(
            result, max_tokens, response_schema, images, detail, **kwargs
        )
        return new_message, result

    def register_plugin(self, plugin: ChatPlugin):
        plugin.Controller = self
        plugin._register()

    def unregister_plugin(self, plugin: ChatPlugin):
        if plugin.Controller is self:
            plugin._unregister()

    def _run_plugins(self, prompt, *args, **kwargs):
        for plugin in self._plugins:
            plugin._pre_chat(prompt, *args, **kwargs)

    async def _arun_plugins(self, prompt, *args, **kwargs):
        for plugin in self._plugins:
            await plugin._pre_achat(prompt, *args, **kwargs)

    def _clean_plugins(self, result, *args, **kwargs):
        for plugin in reversed(self._plugins):
            result = plugin._post_chat(result, *args, **kwargs)
        return result

    async def _aclean_plugins(self, result, *args, **kwargs):
        for plugin in reversed(self._plugins):
            result = await plugin._post_achat(result, *args, **kwargs)
        return result

    def _prechat(self, prompt, max_tokens, images=None, detail="low"):
        images = LLMImage.list_from_images(images, detail=detail)
        if prompt in [LLM_INAPPROPRIATE_QUESTION]:
            return (
                LLMMessage(Role="error", Message=prompt),
                [],
            )
        return self._build_latest_convo(prompt, images, max_tokens)

    def _postchat(self, result):
        if self.Keep_History:
            self.History.append(result)

    def _build_latest_convo(self, prompt, images, max_tokens):
        if any(
            val is not None
            for val in [self.Sys_Msg.get(-1, None), self.Sys_Msg.get(-2, None)]
        ):
            if not self.Caller.Model.Flexible_SysMsg:
                raise ValueError(
                    f"Caller {self.Caller} doesn't support multiple system message placements"
                )

        def build_messagelist():
            sys_msg_llmmsg = {
                idx: LLMMessage(Role="system", Message=msg)
                for idx, msg in self.Sys_Msg.items()
            }
            sys_msg_conf_llmmsg = {
                idx: LLMMessage(Role="system", Message=msg)
                for idx, msg in self.Sys_Msg_Confirmation.items()
            }
            if new_message is not None:
                latest_convo = [*self.History, new_message]
            else:
                latest_convo = [*self.History]
            for idx, msg in sorted(sys_msg_llmmsg.items()):
                if idx in sys_msg_conf_llmmsg:
                    usemsg = [msg, sys_msg_conf_llmmsg[idx]]
                else:
                    usemsg = [msg]
                if idx == -1:
                    latest_convo.extend(usemsg)
                elif idx == -2:
                    latest_convo = [
                        *latest_convo[: idx + 1],
                        *usemsg,
                        *latest_convo[idx + 1 :],
                    ]
                elif idx == 0:
                    latest_convo = [*usemsg, *latest_convo]
            return latest_convo

        if prompt:
            new_message = LLMMessage(Role="user", Message=prompt, Images=images)
        else:
            new_message = None
        latest_convo = build_messagelist()
        x = 1
        while (
            self.Caller.count_tokens(latest_convo)
            > self.Caller.Token_Window - max_tokens
        ):
            if len(self.History) > 0:
                self.History.pop(0)
                latest_convo = build_messagelist()
            else:
                raise LLMError(LLM_QUESTION_TOO_LONG)
        if new_message is not None and self.Keep_History:
            self.History.append(new_message)
        return new_message, latest_convo


class ChatPlugin(ABC, BaseModel):
    Caller: LLMCaller
    Controller: Optional[ChatController] = None
    Sys_Msg: Optional[str] = None
    Restore_Attrs: List[str] = Field(default_factory=list)
    _restore_vals: Dict[str, Any] = PrivateAttr(default_factory=dict)

    @abstractmethod
    def register(self):
        pass

    @abstractmethod
    def unregister(self):
        pass

    @abstractmethod
    def pre_chat(self, prompt: str, *args, **kwargs):
        pass

    @abstractmethod
    async def pre_achat(self, prompt: str, *args, **kwargs):
        pass

    @abstractmethod
    def post_chat(self, result: LLMMessage, *args, **kwargs):
        return result

    @abstractmethod
    async def post_achat(self, result: LLMMessage, *args, **kwargs):
        return result

    def _register(self):
        self._common_register()
        self.register()

    def _unregister(self):
        self._common_unregister()
        self.unregister()

    def _common_register(self):
        if self not in self.Controller._plugins:
            self.Controller._plugins.append(self)

    def _common_unregister(self):
        if self in self.Controller._plugins:
            self.Controller._plugins.remove(self)

    def _pre_chat(self, *args, **kwargs):
        self._common_pre_chat()
        self.pre_chat(*args, **kwargs)

    async def _pre_achat(self, *args, **kwargs):
        self._common_pre_chat()
        await self.pre_achat(*args, **kwargs)

    def _post_chat(self, result, *args, **kwargs):
        result = self.post_chat(result, *args, **kwargs)
        self._common_post_chat()
        return result

    async def _post_achat(self, result, *args, **kwargs):
        result = await self.post_achat(result, *args, **kwargs)
        self._common_post_chat()
        return result

    def _common_pre_chat(self):
        for attr in self.Restore_Attrs:
            obj, *_ = self._obj_from_string(attr)
            self._restore_vals[attr] = copy.deepcopy(obj)

    def _common_post_chat(self):
        for attr in self.Restore_Attrs:
            _, parent, useattr = self._obj_from_string(attr)
            setattr(parent, useattr, self._restore_vals[attr])
        self._restore_vals = {}

    def _obj_from_string(self, string):
        obj_names = string.split(".")
        parent = self.Controller
        obj = getattr(self.Controller, obj_names[0])
        for objn in obj_names[1:]:
            parent = obj
            obj = getattr(obj, objn)
        attr = obj_names[-1]
        return obj, parent, attr


ChatController.model_rebuild()


class LLMError(Exception):
    pass
