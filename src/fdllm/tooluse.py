from __future__ import annotations

import copy
from typing import Optional, List, Literal, Any, Dict, ClassVar
from abc import ABC, abstractmethod

from pydantic import BaseModel
from .llmtypes import LLMCaller, LLMMessage
from .chat import ChatPlugin


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
                raise ValueError(f"Invalid parameter {param}")
            else:
                valid_params[param] = val
        for param, val in self.params.items():
            if val.required and param not in params:
                raise ValueError(f"Required param {param} not present")
        return valid_params

    def dict(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {key: val.dict() for key, val in self.params.items()},
                    "required": [
                        key for key, val in self.params.items() if val.required
                    ],
                },
            },
        }

    model_config = {"arbitrary_types_allowed": True}


class ToolUsePlugin(ChatPlugin):
    Caller: Optional[LLMCaller] = None
    Tools: List[Tool]

    def register(self):
        return super().register()

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
        resp = [
            await self.tool_dict[tc.Name]._aexecute(**tc.Args)
            for tc in result.ToolCalls
        ]
        self._post_chat_appender(resp)
        _, result = await self.Controller.achat("", *args, **kwargs)
        return result

    def post_chat(self, result: LLMMessage, *args, **kwargs):
        self.Controller.Caller.Defaults.pop("tools")
        if result.ToolCalls is None:
            return result
        resp = [self.tool_dict[tc.Name]._execute(**tc.Args) for tc in result.ToolCalls]
        self._post_chat_appender(resp)
        _, result = self.Controller.chat("", *args, **kwargs)
        return result

    async def pre_achat(self, prompt: str, *args, **kwargs):
        return self.pre_chat(prompt, *args, **kwargs)

    def pre_chat(self, prompt: str, *args, **kwargs):
        self.Controller.Caller.Defaults["tools"] = self.dict()

    def dict(self):
        return [tool.dict() for tool in self.Tools]

    @property
    def tool_dict(self):
        return {tool.name: tool for tool in self.Tools}
