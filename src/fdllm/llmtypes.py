from __future__ import annotations
from typing import (
    List,
    Literal,
    Callable,
    Awaitable,
    Any,
    Optional,
    Dict,
    Union,
    Type,
    ClassVar,
)
import datetime
import logging
from abc import ABC, abstractmethod
import os
from dataclasses import field
from functools import wraps
from pathlib import Path
from io import BytesIO
from copy import copy
import base64
import json

import numpy as np
from openai import RateLimitError as RateLimitErrorOpenAI, APIConnectionError
from anthropic import RateLimitError as RateLimitErrorAnthropic
from google.genai.errors import ServerError
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict, BaseModel, Field
from PIL import Image, ImageFile

from .decorators import delayedretry
from .logging_utils import get_logger, log_call_start, log_call_completion

# Tenacity imports for superior retry functionality
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception_type,
    before_log,
    after_log, 
    before_sleep_log
)
from .openai.tokenizer import tokenize_chatgpt_messages
from .constants import LLM_DEFAULT_MAX_TOKENS, LLM_DEFAULT_MAX_RETRIES
from .sysutils import load_models, deepmerge_dicts, get_google_token


class LLMModelType(BaseModel):
    Name: Optional[str]  # why is name optional?
    Api_Interface: str
    Api_Key_Env_Var: Optional[str] = None
    Api_Model_Name: Optional[str] = None
    Max_Token_Arg_Name: str = "max_completion_tokens"
    Token_Window: int
    Token_Limit_Completion: Optional[int] = None
    Client_Args: dict = Field(default_factory=dict)
    Call_Args: dict = Field(default_factory=dict)
    Extra_Body: dict = Field(default_factory=dict)
    Tool_Use: bool = False
    Vision: bool = False
    Flexible_SysMsg: bool = True
    _default_client_args = {}

    def __init__(self, Name):
        models = load_models()
        if Name not in models:
            raise NotImplementedError(
                f"{Name} is not a recognised model name, check models.yaml"
            )

        # initialize pydantic object with the config
        model_config = copy(models[Name])
        super().__init__(Name=Name, **model_config)

        # if no Api_Model_Name is set we use the model name directly
        if self.Api_Model_Name is None:
            self.Api_Model_Name = Name
        # apply defaults from subclass
        self.Client_Args = deepmerge_dicts(
            self._default_client_args, model_config.get("Client_Args", {})
        )
        self._set_api_key_from_env()

    @classmethod
    def model_types(cls) -> Dict[str, Type["LLMModelType"]]:
        return {
            "OpenAI": OpenAIModelType,
            "AzureOpenAI": AzureOpenAIModelType,
            "AzureMistralAI": AzureMistralAIModelType,
            "Anthropic": AnthropicModelType,
            "AnthropicStreaming": AnthropicStreamingModelType,
            "VertexAI": VertexAIModelType,
            "GoogleGenAI": GoogeGenAIModelType,
            "Bedrock": BedrockModelType,
        }

    @classmethod
    def get_type(cls, name) -> LLMModelType:
        models = load_models()
        if name not in models:
            raise NotImplementedError(
                f"{name} is not a recognised model name, check models.yaml"
            )
        MODEL_TYPES = cls.model_types()
        if models[name]["Api_Interface"] not in MODEL_TYPES:
            raise ValueError(
                "Unknown api_interface setting, check models.yaml config file"
            )
        else:
            return MODEL_TYPES[models[name]["Api_Interface"]]

    def _set_api_key_from_env(self):
        if self.Client_Args.get("api_key", None) is None:
            if (self.Api_Key_Env_Var is None) or (
                self.Api_Key_Env_Var not in os.environ
            ):
                raise ValueError(
                    f"{self.Name} does not have api_key or Api_Key_Env_Var set"
                )
            self.Client_Args["api_key"] = os.environ[self.Api_Key_Env_Var]


class OpenAIModelType(LLMModelType):
    Api_Key_Env_Var: str = "OPENAI_API_KEY"


class AzureOpenAIModelType(LLMModelType):
    Api_Key_Env_Var: str = "AZURE_OPENAI_API_KEY"


class AzureMistralAIModelType(LLMModelType):
    Api_Key_Env_Var: str = "MISTRAL_API_KEY"


class AnthropicModelType(LLMModelType):
    Api_Key_Env_Var: str = "ANTHROPIC_API_KEY"

class AnthropicStreamingModelType(LLMModelType):
    Api_Key_Env_Var: str = "ANTHROPIC_API_KEY"

class BedrockModelType(LLMModelType):
    Api_Key_Env_Var: str = "AWS_API_KEYS"

    def _set_api_key_from_env(self):
        if "aws_access_key_id" not in self.Client_Args:
            env_api_key = os.getenv(self.Api_Key_Env_Var)
            aws_access_key_id, aws_secret_access_key = env_api_key.split(" ")
            self.Client_Args["aws_access_key_id"] = aws_access_key_id
            self.Client_Args["aws_secret_access_key"] = aws_secret_access_key


class VertexAIModelType(LLMModelType):

    def _set_api_key_from_env(self):
        if "api_key" not in self.Client_Args:
            self.Client_Args["api_key"] = get_google_token()


class GoogeGenAIModelType(LLMModelType):
    Api_Key_Env_Var: str = "GEMINI_API_KEY"


class LLMMessage(BaseModel):
    Role: Literal["user", "assistant", "system", "tool", "error"]
    Message: Optional[str] = None
    Images: Optional[List[LLMImage]] = None
    ToolCalls: Optional[List[LLMToolCall]] = None
    TokensUsed: int = 0
    DateUTC: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

    def __eq__(self, __value: object) -> bool:
        # exclude timestamp from equality test
        if isinstance(__value, self.__class__):
            return (
                self.Role == __value.Role
                and self.Message == __value.Message
                and self.Images == __value.Images
                and self.TokensUsed == __value.TokensUsed
            )
        else:
            return super().__eq__(__value)

    model_config = {"arbitrary_types_allowed": True}


class LLMToolCall(BaseModel):
    ID: Optional[str]
    Name: str
    Args: dict = Field(default_factory=dict)
    Response: Optional[str] = None


class LLMImage(BaseModel):
    Url: Optional[str] = None
    Img: Optional[Union[Image.Image, Path]] = None
    Detail: Literal["low", "high"] = "low"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._format_image()

    @classmethod
    def list_from_images(
        cls, images: Optional[List[Image.Image]], detail: Literal["low", "high"] = "low"
    ):
        if images is None:
            return
        else:
            return [cls(Img=img, Detail=detail) for img in images]

    def encode(self):
        if self.Img is None:
            return
        img_byte_arr = self.get_bytes()
        return base64.b64encode(img_byte_arr).decode("utf-8")

    def get_bytes(self):
        if self.Img is None:
            return
        img_byte_arr = BytesIO()
        self.Img.convert("RGB").save(img_byte_arr, format="png")
        return img_byte_arr.getvalue()

    def tokenize(self):
        if self.Detail == "low":
            return 85
        else:
            if self.Img is None:
                # if image is url we use worst case scenario
                # for width and height
                width, height = 2048, 768
            else:
                width, height = self.Img.size
            ngridw = int(np.ceil(width / 512))
            ngridh = int(np.ceil(height / 512))
            ntiles = ngridw * ngridh
            return ntiles * 170 + 85

    def _format_image(self):
        im = self.Img
        if im is None:
            return
        detail = self.Detail
        width, height = im.size
        if detail == "low":
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            im = im.resize((512, 512), Image.BILINEAR)
        else:
            maxdim = max(width, height)
            if maxdim > 2048:
                width = width * (2048 / maxdim)
                height = height * (2048 / maxdim)
            shortestdim = min(width, height)
            scale = min(768 / shortestdim, 1)
            finalwidth = int(width * scale)
            finalheight = int(height * scale)
            im = im.resize((finalwidth, finalheight), Image.BILINEAR)
        self.Img = im

    model_config = {"arbitrary_types_allowed": True}


@dataclass(config=ConfigDict(validate_assignment=True))
class LLMCallArgNames:
    Messages: str
    Model: str
    Max_Tokens: str
    Response_Schema: Optional[str] = None


class LLMCaller(ABC):
    def __init__(self, *, Model: LLMModelType, Func: Callable[..., Any], AFunc: Callable[..., Awaitable[Any]], 
                 Token_Window: int, Token_Limit_Completion: Optional[int] = None, 
                 Defaults: Optional[Dict] = None, Arg_Names: Optional[LLMCallArgNames] = None, **kwargs):
        self.Model: LLMModelType = Model
        self.Func: Callable[..., Any] = Func
        self.AFunc: Callable[..., Awaitable[Any]] = AFunc
        self.Token_Window: int = Token_Window
        self.Token_Limit_Completion: Optional[int] = Token_Limit_Completion
        self.Defaults: Dict = Defaults or {}
        self.Arg_Names: Optional[LLMCallArgNames] = Arg_Names
        
        # Thread-safe decorated methods created once during initialization
        self._sync_call_with_retry: Optional[Callable] = None
        self._async_call_with_retry: Optional[Callable] = None
        
        # Handle additional attributes (like Client, AClient for some providers)
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        # Initialize retry methods after all attributes are set
        self._create_retry_methods()
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for this caller."""
        return get_logger(f"caller.{self.Model.Name or 'unknown'}")
    
    def _create_retry_methods(self):
        """Create Tenacity-based retry methods with superior logging."""
        
        # Sync version with Tenacity
        @retry(
            stop=stop_after_attempt(LLM_DEFAULT_MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type((
                RateLimitErrorOpenAI,
                RateLimitErrorAnthropic,
                APIConnectionError,
                ServerError,
            )),
            before=before_log(self.logger, logging.DEBUG),
            before_sleep=before_sleep_log(self.logger, logging.WARNING, exc_info=True),
            after=after_log(self.logger, logging.DEBUG),
            reraise=True
        )
        def sync_retry_wrapper(*args, **kwargs):
            return self.Func(*args, **kwargs)
        
        # Async version with Tenacity  
        @retry(
            stop=stop_after_attempt(LLM_DEFAULT_MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type((
                RateLimitErrorOpenAI,
                RateLimitErrorAnthropic,
                APIConnectionError,
                ServerError,
            )),
            before=before_log(self.logger, logging.DEBUG),
            before_sleep=before_sleep_log(self.logger, logging.WARNING, exc_info=True),
            after=after_log(self.logger, logging.DEBUG),
            reraise=True
        )
        async def async_retry_wrapper(*args, **kwargs):
            return await self.AFunc(*args, **kwargs)
        
        self._sync_call_with_retry = sync_retry_wrapper
        self._async_call_with_retry = async_retry_wrapper

    @abstractmethod
    def format_message(self, message: LLMMessage):
        pass

    @abstractmethod
    def format_messagelist(self, messagelist: List[LLMMessage]):
        pass

    @abstractmethod
    def format_output(
        self, output: Any, response_schema: Optional[BaseModel] = None
    ) -> LLMMessage:
        pass

    def tokenize(self, messagelist: List[LLMMessage]) -> List[int]:
        pass

    def count_tokens(self, messagelist: List[LLMMessage]) -> int:
        return len(self.tokenize(messagelist))

    def format_tool(self, tool):
        pass

    def format_tools(self, tools):
        return [self.format_tool(tool) for tool in tools]

    def sanitize_messagelist(
        self, messagelist: List[LLMMessage], min_new_token_window: int
    ) -> List[LLMMessage]:
        out = messagelist
        while self.Token_Window - self.count_tokens(messagelist) < min_new_token_window:
            out = out[1:]
        return out

    def call(
        self,
        messages: List[LLMMessage] | LLMMessage,
        max_tokens: Optional[int] = LLM_DEFAULT_MAX_TOKENS,
        response_schema: Optional[BaseModel] = None,
        **kwargs,
    ):
        # Ensure messages is a list for logging
        msg_list = messages if isinstance(messages, list) else [messages]
        
        # Log call start
        start_time = log_call_start(self.logger, self.Model.Name, msg_list, "sync")
        
        try:
            kwargs = self._proc_call_args(messages, max_tokens, response_schema, **kwargs)
            response = self._call(**kwargs)
            formatted_output = self.format_output(response, response_schema=response_schema)
            
            # Log successful completion
            log_call_completion(self.logger, self.Model.Name, start_time, response)
            return formatted_output
        except Exception as e:
            # Log failed completion
            log_call_completion(self.logger, self.Model.Name, start_time, error=e)
            raise

    async def acall(
        self,
        messages: List[LLMMessage] | LLMMessage,
        max_tokens: Optional[int] = LLM_DEFAULT_MAX_TOKENS,
        response_schema: Optional[BaseModel] = None,
        **kwargs,
    ):
        # Ensure messages is a list for logging
        msg_list = messages if isinstance(messages, list) else [messages]
        
        # Log call start
        start_time = log_call_start(self.logger, self.Model.Name, msg_list, "async")
        
        try:
            kwargs = self._proc_call_args(messages, max_tokens, response_schema, **kwargs)
            response = await self._acall(**kwargs)
            formatted_output = self.format_output(response, response_schema=response_schema)
            
            # Log successful completion
            log_call_completion(self.logger, self.Model.Name, start_time, response)
            return formatted_output
        except Exception as e:
            # Log failed completion
            log_call_completion(self.logger, self.Model.Name, start_time, error=e)
            raise

    def _proc_call_args(self, messages, max_tokens, response_schema, **kwargs):
        if isinstance(messages, LLMMessage):
            messages = [messages]
        if max_tokens is None:
            max_tokens = self.Token_Window - (self.count_tokens(messages) + 64)
        if self.Token_Limit_Completion is not None:
            max_tokens = min(max_tokens, self.Token_Limit_Completion)
        if self.Arg_Names is not None:
            kwargs[self.Arg_Names.Model] = self.Model.Api_Model_Name
            kwargs[self.Arg_Names.Max_Tokens] = max_tokens
            kwargs[self.Arg_Names.Messages] = self.format_messagelist(messages)
            if (
                self.Arg_Names.Response_Schema is not None
                and response_schema is not None
            ):
                kwargs[self.Arg_Names.Response_Schema] = response_schema
        return {**self.Model.Call_Args, **self.Defaults, **kwargs}

    def _call(self, *args, **kwargs):
        """Thread-safe synchronous call with retry logic."""
        return self._sync_call_with_retry(*args, **kwargs)

    async def _acall(self, *args, **kwargs):
        """Thread-safe asynchronous call with retry logic."""
        return await self._async_call_with_retry(*args, **kwargs)


class LiteralCaller(LLMCaller):
    def __init__(self, text: str):
        # Create a minimal model type for literal caller
        class LiteralModelType:
            Name = "literal"
            Api_Interface = "literal"
            Token_Window = 0
        
        model_type = LiteralModelType()
        
        super().__init__(
            Model=model_type,
            Func=lambda: text,
            AFunc=self._literalafunc(text),
            Token_Window=0,
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


LLMMessage.model_rebuild()
