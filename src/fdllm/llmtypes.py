from __future__ import annotations
from typing import List, Literal, Callable, Awaitable, Any, Optional, Dict
import datetime
from abc import ABC, abstractmethod
import os
from dataclasses import field
from functools import wraps

from openai.error import RateLimitError
from anthropic import ApiException
from anthropic import count_tokens
from anthropic.tokenizer import get_tokenizer
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict, BaseModel, Field

from .decorators import delayedretry
from .openai.tokenizer import tokenize_chatgpt_messages
from .constants import LLM_DEFAULT_MAX_TOKENS, LLM_DEFAULT_MAX_RETRIES

ModelTypeLiteral = Optional[
    Literal[
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613",
        "gpt-4",
        "gpt-4-0314",
        "gpt-4-0613",
        "claude-v1",
        "claude-v1-100k",
        "claude-instant-v1",
        "claude-instant-v1-100k",
        "claude-2",
        "fabdata-openai-devel-gpt4",
        "fabdata-openai-devel-gpt432k",
        "fabdata-openai-devel-gpt35",
    ]
]


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
    Token_Window: int
    APIKey: Optional[str] = None
    Defaults: Dict = Field(default_factory=dict)
    Args: Optional[LLMCallArgs] = None

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
    
    def sanitize_messagelist(
        self, messagelist: List[LLMMessage], min_new_token_window: int
    ) -> List[LLMMessage]:
        out = messagelist
        while (
            self.Token_Window - len(self.tokenize(messagelist)) 
            < min_new_token_window
        ):
            out = out[1:]
        return out

    def call(
        self,
        messages: List[LLMMessage] | LLMMessage,
        max_tokens: Optional[int] = LLM_DEFAULT_MAX_TOKENS,
        **kwargs,
    ):
        kwargs = self._proc_call_args(messages, max_tokens, **kwargs)
        return self.format_output(self._call(**kwargs))

    async def acall(
        self,
        messages: List[LLMMessage] | LLMMessage,
        max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
        **kwargs,
    ):
        kwargs = self._proc_call_args(messages, max_tokens, **kwargs)
        return self.format_output(await self._acall(**kwargs))

    def _proc_call_args(self, messages, max_tokens, **kwargs):
        if isinstance(messages, LLMMessage):
            messages = [messages]
        if max_tokens is None:
            max_tokens = self.Token_Window - (len(self.tokenize(messages)) + 64)
        if self.Args is not None:
            kwargs[self.Args.Model] = self.Model.Name
            kwargs[self.Args.Max_Tokens] = max_tokens
            kwargs[self.Args.Messages] = self.format_messagelist(messages)
        return {**self.Defaults, **kwargs}

    @delayedretry(
        rethrow_final_error=True,
        max_attempts=LLM_DEFAULT_MAX_RETRIES,
        include_errors=[RateLimitError, ApiException]
    )
    def _call(self, *args, **kwargs):
        return self.Func(*args, **kwargs)

    @delayedretry(
        rethrow_final_error=True,
        max_attempts=LLM_DEFAULT_MAX_RETRIES,
        include_errors=[RateLimitError, ApiException]
    )
    async def _acall(self, *args, **kwargs):
        return await self.AFunc(*args, **kwargs)


class LiteralCaller(LLMCaller):
    def __init__(self, text: str):
        super().__init__(
            Model = LLMModelType(Name=None),
            Func = lambda: text,
            AFunc = self._literalafunc(text),
            Token_Window = 0,
        )

    @staticmethod
    def _literalafunc(text):
        async def afunc():
            return text
        return afunc

    def format_message(self, message: LLMMessage):
        return super().format_message(message)

    def format_messagelist(self, messagelist: List[LLMMessage]):
        return super().format_messagelist(messagelist)

    def format_output(self, output: Any) -> LLMMessage:
        return LLMMessage(Role="assistant", Message=output)

    def tokenize(self, messagelist: List[LLMMessage]) -> List[int]:
        return super().tokenize(messagelist)