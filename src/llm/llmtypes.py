from __future__ import annotations
from typing import (
    List, Literal, Callable, Awaitable, Any, Tuple
)
import datetime
from abc import ABC, abstractmethod
import os
from dataclasses import field
from functools import wraps

import openai
import anthropic
from anthropic import count_tokens
from anthropic.tokenizer import get_tokenizer
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict, BaseModel

from .decorators import delayedretry
from .openai.tokenizer import tokenize_chatgpt_messages
from .constants import LLM_DEFAULT_MAX_TOKENS

ModelTypeLiteral = Literal["gpt-3.5-turbo", "gpt-4", "claude-v1"]

@dataclass(config=ConfigDict(validate_assignment=True))
class LLMModelType:
    Name: ModelTypeLiteral


@dataclass(config=ConfigDict(validate_assignment=True))
class LLMMessage:
    Role: Literal["user", "assistant", "system", "error"]
    Message: str
    TokensUsed: int = 0
    DateUTC: datetime.datetime = datetime.datetime.utcnow()


@dataclass(config=ConfigDict(validate_assignment=True))
class LLMCallArgs:
    Messages: str
    Model: str
    Max_Tokens: str


class LLMCaller(ABC, BaseModel):
    Model: LLMModelType
    Func: Callable[..., Any]
    AFunc: Callable[..., Awaitable[Any]]
    Args: LLMCallArgs
    APIKey: str
    Defaults: dict
    Token_Window: int
    
    @abstractmethod
    def format_message(self, message: LLMMessage):
        pass

    @abstractmethod
    def format_messagelist(self, messagelist: List[LLMMessage]):
        pass
    
    @abstractmethod
    def format_output(self, output: Any) -> LLMMessage:
        pass

    @abstractmethod
    def tokenize(self, messagelist: List[LLMMessage]) -> List[int]:
        pass

    def call(
        self,
        messages: List[LLMMessage] | LLMMessage,
        max_tokens: int=LLM_DEFAULT_MAX_TOKENS,
        **kwargs
    ):
        kwargs = self._proc_call_args(messages, max_tokens, **kwargs)
        return self.format_output(self._call(**kwargs))
    
    async def acall(
        self,
        messages: List[LLMMessage] | LLMMessage,
        max_tokens: int=LLM_DEFAULT_MAX_TOKENS,
        **kwargs
    ):
        kwargs = self._proc_call_args(messages, max_tokens, **kwargs)
        return self.format_output(await self._acall(**kwargs))

    def _proc_call_args(self, messages, max_tokens, **kwargs):
        kwargs[self.Args.Model] = self.Model.Name
        kwargs[self.Args.Max_Tokens] = max_tokens
        if isinstance(messages, LLMMessage):
            messages = [messages]
        kwargs[self.Args.Messages] = self.format_messagelist(messages)
        return {**self.Defaults, **kwargs}

    @delayedretry(rethrow_final_error=True)
    def _call(self, *args, **kwargs):
        return self.Func(*args, **kwargs)

    @delayedretry(rethrow_final_error=True)
    async def _acall(self, *args, **kwargs):
        return await self.AFunc(*args, **kwargs)
    