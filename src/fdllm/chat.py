from __future__ import annotations
from typing import List, Literal, Optional, Any, Dict
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
from .prompt_helpers import (
    clean_prompt_string,
    build_prompt_string,
    contains_banned_words,
)
from .llmtypes import LLMMessage, LLMCaller, LLMImage


class ChatController(BaseModel):
    Caller: LLMCaller
    History: List[LLMMessage] = Field(default_factory=list)
    Sys_Msg: Dict[Literal[0, -1, -2], str] = Field(default_factory=dict)
    Sys_Msg_Confirmation: Dict[Literal[0, -1, -2], str] = Field(default_factory=dict)
    Clean_Prompt: bool = True
    Allow_Banned: bool = False
    Keep_History: bool = True
    _plugins: List[ChatPlugin] = PrivateAttr(default_factory=list)

    def chat(
        self,
        prompt: str,
        max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
        images: Optional[List[Image.Image]] = None,
        detail: Literal["low", "high"] = "low",
        **kwargs,
    ):
        self._run_plugins(prompt)
        try:
            new_message, latest_convo = self._prechat(
                prompt, max_tokens, images, detail
            )
        except:
            self._clean_plugins()
            raise
        if new_message.Role == "error":
            self._clean_plugins()
            return new_message, None
        result = self.Caller.call(latest_convo, max_tokens, **kwargs)
        self._postchat(result)
        self._clean_plugins()
        return new_message, result

    async def achat(
        self,
        prompt: str,
        max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
        images: Optional[List[Image.Image]] = None,
        detail: Literal["low", "high"] = "low",
        **kwargs,
    ):
        await self._arun_plugins(prompt)
        try:
            new_message, latest_convo = self._prechat(
                prompt, max_tokens, images, detail
            )
        except:
            await self._aclean_plugins()
            raise
        if new_message.Role == "error":
            await self._aclean_plugins()
            return new_message, None
        result = await self.Caller.acall(latest_convo, max_tokens, **kwargs)
        self._postchat(result)
        await self._aclean_plugins()
        return new_message, result

    def register_plugin(self, plugin: ChatPlugin):
        plugin.Controller = self
        plugin._register()

    def unregister_plugin(self, plugin: ChatPlugin):
        if plugin.Controller is self:
            plugin._unregister()

    def _run_plugins(self, prompt):
        for plugin in self._plugins:
            plugin._pre_chat(prompt)

    async def _arun_plugins(self, prompt):
        for plugin in self._plugins:
            await plugin._pre_achat(prompt)

    def _clean_plugins(self):
        for plugin in reversed(self._plugins):
            plugin._post_chat()

    async def _aclean_plugins(self):
        for plugin in reversed(self._plugins):
            await plugin._post_achat()

    def _prechat(self, prompt, max_tokens, images=None, detail="low"):
        images = LLMImage.list_from_images(images, detail=detail)
        final_prompt = self._clean(prompt)
        if final_prompt in [LLM_EMPTY_QUESTION, LLM_INAPPROPRIATE_QUESTION]:
            return (
                LLMMessage(Role="error", Message=final_prompt),
                [],
            )
        return self._build_latest_convo(final_prompt, images, max_tokens)

    def _postchat(self, result):
        if self.Keep_History:
            self.History.append(result)

    def _clean(self, prompt):
        if self.Clean_Prompt:
            final_prompt = clean_prompt_string(build_prompt_string(prompt))
        else:
            final_prompt = prompt
        if len(final_prompt) == 0:
            return LLM_EMPTY_QUESTION
        if not self.Allow_Banned and contains_banned_words(final_prompt):
            return LLM_INAPPROPRIATE_QUESTION
        return final_prompt

    def _build_latest_convo(self, prompt, images, max_tokens):
        def build_messagelist():
            sys_msg_llmmsg = {
                idx: LLMMessage(Role="system", Message=msg)
                for idx, msg in self.Sys_Msg.items()
            }
            sys_msg_conf_llmmsg = {
                idx: LLMMessage(Role="system", Message=msg)
                for idx, msg in self.Sys_Msg_Confirmation.items()
            }
            latest_convo = [*self.History, new_message]
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

        new_message = LLMMessage(Role="user", Message=prompt, Images=images)
        latest_convo = build_messagelist()
        while (
            len(self.Caller.tokenize(latest_convo))
            > self.Caller.Token_Window - max_tokens
        ):
            if len(self.History) > 0:
                self.History.pop(0)
                latest_convo = build_messagelist()
            else:
                raise LLMError(LLM_QUESTION_TOO_LONG)
        if self.Keep_History:
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
    def pre_chat(self, prompt: str):
        pass

    @abstractmethod
    async def pre_achat(self, prompt: str):
        pass

    @abstractmethod
    def post_chat(self):
        pass

    @abstractmethod
    async def post_achat(self):
        pass

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

    def _pre_chat(self, prompt):
        self._common_pre_chat()
        self.pre_chat(prompt)

    async def _pre_achat(self, prompt):
        self._common_pre_chat()
        await self.pre_achat(prompt)

    def _post_chat(self):
        self.post_chat()
        self._common_post_chat()

    async def _post_achat(self):
        await self.post_achat()
        self._common_post_chat()

    def _common_pre_chat(self):
        for attr in self.Restore_Attrs:
            self._restore_vals[attr] = copy.deepcopy(getattr(self.Controller, attr))

    def _common_post_chat(self):
        for attr in self.Restore_Attrs:
            setattr(self.Controller, attr, self._restore_vals[attr])
        self._restore_vals = {}

    class Config:
        underscore_attrs_are_private = True


ChatController.update_forward_refs()


class LLMError(Exception):
    pass
