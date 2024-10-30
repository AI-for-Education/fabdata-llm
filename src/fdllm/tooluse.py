from __future__ import annotations

import copy
from typing import Optional, List, Literal, Any, Dict, ClassVar
from abc import ABC, abstractmethod

from pydantic import BaseModel
from .llmtypes import (
    LLMCaller,
    LLMMessage,
    LLMModelType,
    OpenAIModelType,
    AnthropicModelType,
    AzureOpenAIModelType,
    AzureMistralAIModelType,
)
from .chat import ChatPlugin


class ToolMissingParamError(Exception):
    pass


class ToolInvalidParamError(Exception):
    pass


class _ToolParamBase(BaseModel):
    type: Literal["string", "array", "integer", "number", "boolean", "null"]
    items: Optional[ToolItem] = None
    enum: Optional[List[Any]] = None
    description: Optional[str] = None
    default: Optional[Any] = None

    def dict(self):
        outdict = {"type": self.type}
        if self.description is not None:
            outdict["description"] = self.description
        if self.items is not None:
            outdict["items"] = self.items.dict()
        if self.enum is not None:
            outdict["enum"] = self.enum
        return outdict


class ToolParam(_ToolParamBase):
    required: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.required and self.default is None:
            raise ValueError("Must provide default for non-required param")


class ToolItem(_ToolParamBase):
    pass


class Tool(ABC, BaseModel):
    name: ClassVar[str]
    description: ClassVar[str]
    params: ClassVar[Dict[str, ToolParam]]

    @abstractmethod
    def execute(self, **params):
        pass

    @abstractmethod
    async def aexecute(self, **params):
        pass

    def _execute(self, **params):
        return self.execute(**self._validate_params(**params))

    async def _aexecute(self, **params):
        return await self.aexecute(**self._validate_params(**params))

    def _validate_params(self, **params):
        valid_params = {}
        for param, val in params.items():
            if param not in self.params:
                raise ToolInvalidParamError(f"Invalid parameter {param}")
            else:
                valid_params[param] = val
        for param, val in self.params.items():
            if val.required and param not in params:
                raise ToolMissingParamError(f"Required param {param} not present")
        return valid_params

    def dict(self, model="gpt-4-1106-preview"):
        Modtype = LLMModelType.get_type(model)
        if Modtype in (
            OpenAIModelType,
            AzureOpenAIModelType,
            AzureMistralAIModelType,
        ):
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            key: val.dict() for key, val in self.params.items()
                        },
                        "required": [
                            key for key, val in self.params.items() if val.required
                        ],
                    },
                },
            }
        elif Modtype in (AnthropicModelType,):
            return {
                "name": self.name,
                "description": self.description,
                "input_schema": {
                    "type": "object",
                    "properties": {key: val.dict() for key, val in self.params.items()},
                    "required": [
                        key for key, val in self.params.items() if val.required
                    ],
                },
            }

    model_config = {"arbitrary_types_allowed": True}


class ToolUsePlugin(ChatPlugin):
    Caller: Optional[LLMCaller] = None
    Tools: List[Tool]
    _tool_attempt: int = 0
    _max_tool_attempt: int = 5

    def register(self):
        super().register()
        model = self.Controller.Caller.Model.Name
        modeltype = LLMModelType.get_type(model)(model)
        if not modeltype.Tool_Use:
            raise NotImplementedError(f"{model} doesn''t support tool use")

    def unregister(self):
        return super().unregister()

    def _post_chat_appender(self, resp):
        controller = self.Controller
        tc = copy.deepcopy(controller.History[-1].ToolCalls)
        for i, resp_ in enumerate(resp):
            tc[i].Response = resp_
        controller.History.append(LLMMessage(Role="tool", ToolCalls=tc))

    async def post_achat(self, result: LLMMessage, *args, **kwargs):
        self.Controller.Caller.Defaults.pop("tools")
        if result.ToolCalls is None:
            return result
        try:
            resp = [
                await self.tool_dict[tc.Name]._aexecute(**tc.Args)
                for tc in result.ToolCalls
            ]
        except:
            self._tool_attempt += 1
            self.Controller.History.pop()
            prompt = self.Controller.History.pop()
            if self._tool_attempt > self._max_tool_attempt:
                raise
            _, result = await self.Controller.achat(prompt.Message, *args, **kwargs)
            self._tool_attempt = 0
            return result
        self._post_chat_appender(resp)
        _, result = await self.Controller.achat("", *args, **kwargs)
        self._tool_attempt = 0
        return result

    def post_chat(self, result: LLMMessage, *args, **kwargs):
        self.Controller.Caller.Defaults.pop("tools")
        if result.ToolCalls is None:
            return result
        try:
            resp = [
                self.tool_dict[tc.Name]._execute(**tc.Args) for tc in result.ToolCalls
            ]
        except:
            self._tool_attempt += 1
            self.Controller.History.pop()
            prompt = self.Controller.History.pop()
            if self._tool_attempt > self._max_tool_attempt:
                self._tool_attempt = 0
                raise
            _, result = self.Controller.chat(prompt.Message, *args, **kwargs)
            self._tool_attempt = 0
            return result
        self._post_chat_appender(resp)
        _, result = self.Controller.chat("", *args, **kwargs)
        self._tool_attempt = 0
        return result

    async def pre_achat(self, prompt: str, *args, **kwargs):
        return self.pre_chat(prompt, *args, **kwargs)

    def pre_chat(self, prompt: str, *args, **kwargs):
        self.Controller.Caller.Defaults["tools"] = self.dict(
            self.Controller.Caller.Model.Name
        )

    def dict(self, model="gpt-4-1106-preview"):
        return [tool.dict(model) for tool in self.Tools]

    @property
    def tool_dict(self):
        return {tool.name: tool for tool in self.Tools}
