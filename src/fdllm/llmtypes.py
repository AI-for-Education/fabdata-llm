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
import time
import datetime
import logging
from abc import ABC, abstractmethod
import os
import warnings
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


def _warn_titlecase(attr: str):
    warnings.warn(
        f"Attribute '{attr}' is deprecated; use snake_case instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class LLMModelType(BaseModel):
    name: Optional[str] = Field(alias="Name")  # why is name optional?
    api_interface: str = Field(alias="Api_Interface")
    api_key_env_var: Optional[str] = Field(default=None, alias="Api_Key_Env_Var")
    api_model_name: Optional[str] = Field(default=None, alias="Api_Model_Name")
    max_token_arg_name: str = Field(default="max_completion_tokens", alias="Max_Token_Arg_Name")
    token_window: int = Field(alias="Token_Window")
    token_limit_completion: Optional[int] = Field(default=None, alias="Token_Limit_Completion")
    client_args: dict = Field(default_factory=dict, alias="Client_Args")
    call_args: dict = Field(default_factory=dict, alias="Call_Args")
    extra_body: dict = Field(default_factory=dict, alias="Extra_Body")
    tool_use: bool = Field(default=False, alias="Tool_Use")
    vision: bool = Field(default=False, alias="Vision")
    flexible_sysmsg: bool = Field(default=True, alias="Flexible_SysMsg")
    _default_client_args = {}
    model_config = ConfigDict(populate_by_name=True, validate_assignment=True)

    def __init__(self, name=None, **kwargs):
        if name is None and "Name" in kwargs:
            name = kwargs.pop("Name")
        models = load_models()
        if name not in models:
            raise NotImplementedError(
                f"{name} is not a recognised model name, check models.yaml"
            )

        # initialize pydantic object with the config
        model_config = copy(models[name])
        super().__init__(name=name, **model_config, **kwargs)

        # if no Api_Model_Name is set we use the model name directly
        if self.api_model_name is None:
            self.api_model_name = name
        # apply defaults from subclass
        self.client_args = deepmerge_dicts(
            self._default_client_args, model_config.get("Client_Args", {})
        )
        self._set_api_key_from_env()

    # Snake-case compatibility properties (deprecated TitleCase access remains)
    @property
    def Name(self) -> Optional[str]:
        _warn_titlecase("Name")
        return self.name

    @Name.setter
    def Name(self, value: Optional[str]):
        _warn_titlecase("Name")
        self.name = value

    @property
    def Api_Interface(self) -> str:
        _warn_titlecase("Api_Interface")
        return self.api_interface

    @Api_Interface.setter
    def Api_Interface(self, value: str):
        _warn_titlecase("Api_Interface")
        self.api_interface = value

    @property
    def Api_Key_Env_Var(self) -> Optional[str]:
        _warn_titlecase("Api_Key_Env_Var")
        return self.api_key_env_var

    @Api_Key_Env_Var.setter
    def Api_Key_Env_Var(self, value: Optional[str]):
        _warn_titlecase("Api_Key_Env_Var")
        self.api_key_env_var = value

    @property
    def Api_Model_Name(self) -> Optional[str]:
        _warn_titlecase("Api_Model_Name")
        return self.api_model_name

    @Api_Model_Name.setter
    def Api_Model_Name(self, value: Optional[str]):
        _warn_titlecase("Api_Model_Name")
        self.api_model_name = value

    @property
    def Max_Token_Arg_Name(self) -> str:
        _warn_titlecase("Max_Token_Arg_Name")
        return self.max_token_arg_name

    @Max_Token_Arg_Name.setter
    def Max_Token_Arg_Name(self, value: str):
        _warn_titlecase("Max_Token_Arg_Name")
        self.max_token_arg_name = value

    @property
    def Token_Window(self) -> int:
        _warn_titlecase("Token_Window")
        return self.token_window

    @Token_Window.setter
    def Token_Window(self, value: int):
        _warn_titlecase("Token_Window")
        self.token_window = value

    @property
    def Token_Limit_Completion(self) -> Optional[int]:
        _warn_titlecase("Token_Limit_Completion")
        return self.token_limit_completion

    @Token_Limit_Completion.setter
    def Token_Limit_Completion(self, value: Optional[int]):
        _warn_titlecase("Token_Limit_Completion")
        self.token_limit_completion = value

    @property
    def Client_Args(self) -> dict:
        _warn_titlecase("Client_Args")
        return self.client_args

    @Client_Args.setter
    def Client_Args(self, value: dict):
        _warn_titlecase("Client_Args")
        self.client_args = value

    @property
    def Call_Args(self) -> dict:
        _warn_titlecase("Call_Args")
        return self.call_args

    @Call_Args.setter
    def Call_Args(self, value: dict):
        _warn_titlecase("Call_Args")
        self.call_args = value

    @property
    def Extra_Body(self) -> dict:
        _warn_titlecase("Extra_Body")
        return self.extra_body

    @Extra_Body.setter
    def Extra_Body(self, value: dict):
        _warn_titlecase("Extra_Body")
        self.extra_body = value

    @property
    def Tool_Use(self) -> bool:
        _warn_titlecase("Tool_Use")
        return self.tool_use

    @Tool_Use.setter
    def Tool_Use(self, value: bool):
        _warn_titlecase("Tool_Use")
        self.tool_use = value

    @property
    def Vision(self) -> bool:
        _warn_titlecase("Vision")
        return self.vision

    @Vision.setter
    def Vision(self, value: bool):
        _warn_titlecase("Vision")
        self.vision = value

    @property
    def Flexible_SysMsg(self) -> bool:
        _warn_titlecase("Flexible_SysMsg")
        return self.flexible_sysmsg

    @Flexible_SysMsg.setter
    def Flexible_SysMsg(self, value: bool):
        _warn_titlecase("Flexible_SysMsg")
        self.flexible_sysmsg = value


    @classmethod
    def model_types(cls) -> Dict[str, Type["LLMModelType"]]:
        return {
            "OpenAI": OpenAIModelType,
            "OpenAICompletions": OpenAICompletionsModelType,
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
        model_cfg = models[name]
        api_interface = model_cfg.get("api_interface") or model_cfg.get("Api_Interface")
        MODEL_TYPES = cls.model_types()
        if api_interface not in MODEL_TYPES:
            raise ValueError(
                f"Unknown api_interface setting {api_interface}, check models.yaml config file"
            )
        else:
            return MODEL_TYPES[api_interface]

    def _set_api_key_from_env(self):
        if self.client_args.get("api_key", None) is None:
            if (self.api_key_env_var is None) or (
                self.api_key_env_var not in os.environ
            ):
                raise ValueError(
                    f"{self.name} does not have api_key or Api_Key_Env_Var set"
                )
            self.client_args["api_key"] = os.environ[self.api_key_env_var]


class OpenAIModelType(LLMModelType):
    Api_Key_Env_Var: str = "OPENAI_API_KEY"


class OpenAICompletionsModelType(LLMModelType):
    Api_Key_Env_Var: str = "OPENAI_API_KEY"
    Max_Token_Arg_Name: str = "max_tokens"


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
        if "aws_access_key_id" not in self.client_args:
            env_api_key = os.getenv(self.api_key_env_var)
            aws_access_key_id, aws_secret_access_key = env_api_key.split(" ")
            self.client_args["aws_access_key_id"] = aws_access_key_id
            self.client_args["aws_secret_access_key"] = aws_secret_access_key


class VertexAIModelType(LLMModelType):

    def _set_api_key_from_env(self):
        if "api_key" not in self.client_args:
            self.client_args["api_key"] = get_google_token()


class GoogeGenAIModelType(LLMModelType):
    Api_Key_Env_Var: str = "GEMINI_API_KEY"


class LLMMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool", "error"] = Field(alias="Role")
    message: Optional[str] = Field(default=None, alias="Message")
    images: Optional[List[LLMImage]] = Field(default=None, alias="Images")
    tool_calls: Optional[List[LLMToolCall]] = Field(default=None, alias="ToolCalls")
    tokens_used: Optional[int] = Field(default=None, alias="TokensUsed")
    tokens_used_completion: Optional[int] = Field(default=None, alias="TokensUsedCompletion")
    tokens_used_reasoning: Optional[int] = Field(default=None, alias="TokensUsedReasoning")
    log_probs: Optional[Any] = Field(default=None, alias="LogProbs")
    latency: Optional[float] = Field(default=None, alias="Latency")
    date_utc: datetime.datetime = Field(default_factory=datetime.datetime.utcnow, alias="DateUTC")

    def __eq__(self, __value: object) -> bool:
        # exclude timestamp from equality test
        if isinstance(__value, self.__class__):
            return (
                self.role == __value.role
                and self.message == __value.message
                and self.images == __value.images
                and self.tokens_used == __value.tokens_used
            )
        else:
            return super().__eq__(__value)

    model_config = {"arbitrary_types_allowed": True, "populate_by_name": True}

    # Snake-case compatibility properties
    @property
    def Role(self):
        _warn_titlecase("Role")
        return self.role

    @Role.setter
    def Role(self, value):
        _warn_titlecase("Role")
        self.role = value

    @property
    def Message(self):
        _warn_titlecase("Message")
        return self.message

    @Message.setter
    def Message(self, value):
        _warn_titlecase("Message")
        self.message = value

    @property
    def Images(self):
        _warn_titlecase("Images")
        return self.images

    @Images.setter
    def Images(self, value):
        _warn_titlecase("Images")
        self.images = value

    @property
    def ToolCalls(self):
        _warn_titlecase("ToolCalls")
        return self.tool_calls

    @ToolCalls.setter
    def ToolCalls(self, value):
        _warn_titlecase("ToolCalls")
        self.tool_calls = value

    @property
    def TokensUsed(self):
        _warn_titlecase("TokensUsed")
        return self.tokens_used

    @TokensUsed.setter
    def TokensUsed(self, value):
        _warn_titlecase("TokensUsed")
        self.tokens_used = value

    @property
    def TokensUsedCompletion(self):
        _warn_titlecase("TokensUsedCompletion")
        return self.tokens_used_completion

    @TokensUsedCompletion.setter
    def TokensUsedCompletion(self, value):
        _warn_titlecase("TokensUsedCompletion")
        self.tokens_used_completion = value

    @property
    def TokensUsedReasoning(self):
        _warn_titlecase("TokensUsedReasoning")
        return self.tokens_used_reasoning

    @TokensUsedReasoning.setter
    def TokensUsedReasoning(self, value):
        _warn_titlecase("TokensUsedReasoning")
        self.tokens_used_reasoning = value

    @property
    def LogProbs(self):
        _warn_titlecase("LogProbs")
        return self.log_probs

    @LogProbs.setter
    def LogProbs(self, value):
        _warn_titlecase("LogProbs")
        self.log_probs = value

    @property
    def Latency(self):
        _warn_titlecase("Latency")
        return self.latency

    @Latency.setter
    def Latency(self, value):
        _warn_titlecase("Latency")
        self.latency = value

    @property
    def DateUTC(self):
        _warn_titlecase("DateUTC")
        return self.date_utc

    @DateUTC.setter
    def DateUTC(self, value):
        _warn_titlecase("DateUTC")
        self.date_utc = value


class LLMToolCall(BaseModel):
    id: Optional[str] = Field(alias="ID")
    name: str = Field(alias="Name")
    args: dict = Field(default_factory=dict, alias="Args")
    response: Optional[str] = Field(default=None, alias="Response")

    model_config = {"populate_by_name": True}

    @property
    def ID(self) -> Optional[str]:
        _warn_titlecase("ID")
        return self.id

    @ID.setter
    def ID(self, value: Optional[str]):
        _warn_titlecase("ID")
        self.id = value

    @property
    def Name(self) -> str:
        _warn_titlecase("Name")
        return self.name

    @Name.setter
    def Name(self, value: str):
        _warn_titlecase("Name")
        self.name = value

    @property
    def Args(self) -> dict:
        _warn_titlecase("Args")
        return self.args

    @Args.setter
    def Args(self, value: dict):
        _warn_titlecase("Args")
        self.args = value

    @property
    def Response(self) -> Optional[str]:
        _warn_titlecase("Response")
        return self.response

    @Response.setter
    def Response(self, value: Optional[str]):
        _warn_titlecase("Response")
        self.response = value


class LLMImage(BaseModel):
    url: Optional[str] = Field(default=None, alias="Url")
    img: Optional[Union[Image.Image, Path]] = Field(default=None, alias="Img")
    detail: Literal["low", "high"] = Field(default="low", alias="Detail")

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
            return [cls(img=img, detail=detail) for img in images]

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
        detail = self.detail
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
        self.img = im

    model_config = {"arbitrary_types_allowed": True, "populate_by_name": True}

    @property
    def Url(self) -> Optional[str]:
        _warn_titlecase("Url")
        return self.url

    @Url.setter
    def Url(self, value: Optional[str]):
        _warn_titlecase("Url")
        self.url = value

    @property
    def Img(self) -> Optional[Union[Image.Image, Path]]:
        _warn_titlecase("Img")
        return self.img

    @Img.setter
    def Img(self, value: Optional[Union[Image.Image, Path]]):
        _warn_titlecase("Img")
        self.img = value

    @property
    def Detail(self) -> Literal["low", "high"]:
        _warn_titlecase("Detail")
        return self.detail

    @Detail.setter
    def Detail(self, value: Literal["low", "high"]):
        _warn_titlecase("Detail")
        self.detail = value


class LLMCallArgNames(BaseModel):
    messages: str = Field(alias="Messages")
    model: str = Field(alias="Model")
    max_tokens: str = Field(alias="Max_Tokens")
    response_schema: Optional[str] = Field(default=None, alias="Response_Schema")
    model_config = ConfigDict(validate_assignment=True, populate_by_name=True)

    @property
    def Messages(self) -> str:
        _warn_titlecase("Messages")
        return self.messages

    @Messages.setter
    def Messages(self, value: str):
        _warn_titlecase("Messages")
        self.messages = value

    @property
    def Model(self) -> str:
        _warn_titlecase("Model")
        return self.model

    @Model.setter
    def Model(self, value: str):
        _warn_titlecase("Model")
        self.model = value

    @property
    def Max_Tokens(self) -> str:
        _warn_titlecase("Max_Tokens")
        return self.max_tokens

    @Max_Tokens.setter
    def Max_Tokens(self, value: str):
        _warn_titlecase("Max_Tokens")
        self.max_tokens = value

    @property
    def Response_Schema(self) -> Optional[str]:
        _warn_titlecase("Response_Schema")
        return self.response_schema

    @Response_Schema.setter
    def Response_Schema(self, value: Optional[str]):
        _warn_titlecase("Response_Schema")
        self.response_schema = value


class LLMCaller(ABC, BaseModel):
    model: LLMModelType = Field(alias="Model")
    func: Callable[..., Any] = Field(alias="Func")
    afunc: Callable[..., Awaitable[Any]] = Field(alias="AFunc")
    token_window: int = Field(alias="Token_Window")
    token_limit_completion: Optional[int] = Field(default=None, alias="Token_Limit_Completion")
    defaults: Dict = Field(default_factory=dict, alias="Defaults")
    arg_names: Optional[LLMCallArgNames] = Field(default=None, alias="Arg_Names")
    
    # Thread-safe decorated methods created once during initialization
    _sync_call_with_retry: Optional[Callable] = None
    _async_call_with_retry: Optional[Callable] = None
    model_config = {"populate_by_name": True}

    @property
    def Model(self) -> LLMModelType:
        _warn_titlecase("Model")
        return self.model

    @Model.setter
    def Model(self, value: LLMModelType):
        _warn_titlecase("Model")
        self.model = value

    @property
    def Func(self) -> Callable[..., Any]:
        _warn_titlecase("Func")
        return self.func

    @Func.setter
    def Func(self, value: Callable[..., Any]):
        _warn_titlecase("Func")
        self.func = value

    @property
    def AFunc(self) -> Callable[..., Awaitable[Any]]:
        _warn_titlecase("AFunc")
        return self.afunc

    @AFunc.setter
    def AFunc(self, value: Callable[..., Awaitable[Any]]):
        _warn_titlecase("AFunc")
        self.afunc = value

    @property
    def Token_Window(self) -> int:
        _warn_titlecase("Token_Window")
        return self.token_window

    @Token_Window.setter
    def Token_Window(self, value: int):
        _warn_titlecase("Token_Window")
        self.token_window = value

    @property
    def Token_Limit_Completion(self) -> Optional[int]:
        _warn_titlecase("Token_Limit_Completion")
        return self.token_limit_completion

    @Token_Limit_Completion.setter
    def Token_Limit_Completion(self, value: Optional[int]):
        _warn_titlecase("Token_Limit_Completion")
        self.token_limit_completion = value

    @property
    def Defaults(self) -> Dict:
        _warn_titlecase("Defaults")
        return self.defaults

    @Defaults.setter
    def Defaults(self, value: Dict):
        _warn_titlecase("Defaults")
        self.defaults = value

    @property
    def Arg_Names(self) -> Optional[LLMCallArgNames]:
        _warn_titlecase("Arg_Names")
        return self.arg_names

    @Arg_Names.setter
    def Arg_Names(self, value: Optional[LLMCallArgNames]):
        _warn_titlecase("Arg_Names")
        self.arg_names = value

    def model_post_init(self, __context: Any) -> None:
        """Initialize decorated retry methods after model creation."""
        super().model_post_init(__context) if hasattr(super(), 'model_post_init') else None
        self._create_retry_methods()
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for this caller."""
        return get_logger(f"caller.{self.model.name or 'unknown'}")
    
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
            start_time = time.perf_counter()
            out = self.Func(*args, **kwargs)
            latency = time.perf_counter() - start_time
            return out, latency
        
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
            start_time = time.perf_counter()
            out = await self.AFunc(*args, **kwargs)
            latency = time.perf_counter() - start_time
            return out, latency
        
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
        while self.token_window - self.count_tokens(messagelist) < min_new_token_window:
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
        start_time = log_call_start(self.logger, self.model.name, msg_list, "sync")
        
        try:
            kwargs = self._proc_call_args(messages, max_tokens, response_schema, **kwargs)
            response, latency = self._call(**kwargs)
            formatted_output = self.format_output(response, response_schema=response_schema, latency=latency)
            
            # Log successful completion
            log_call_completion(self.logger, self.model.name, start_time, response)
            return formatted_output
        except Exception as e:
            # Log failed completion
            log_call_completion(self.logger, self.model.name, start_time, error=e)
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
        start_time = log_call_start(self.logger, self.model.name, msg_list, "async")
        
        try:
            kwargs = self._proc_call_args(messages, max_tokens, response_schema, **kwargs)
            response, latency = await self._acall(**kwargs)
            formatted_output = self.format_output(response, response_schema=response_schema, latency=latency)
            
            # Log successful completion
            log_call_completion(self.logger, self.model.name, start_time, response)
            return formatted_output
        except Exception as e:
            # Log failed completion
            log_call_completion(self.logger, self.model.name, start_time, error=e)
            raise

    def _proc_call_args(self, messages, max_tokens, response_schema, **kwargs):
        if isinstance(messages, LLMMessage):
            messages = [messages]
        if max_tokens is None:
            max_tokens = self.token_window - (self.count_tokens(messages) + 64)
        if self.token_limit_completion is not None:
            max_tokens = min(max_tokens, self.token_limit_completion)
        if self.arg_names is not None:
            kwargs[self.arg_names.model] = self.model.api_model_name
            kwargs[self.arg_names.max_tokens] = max_tokens
            kwargs[self.arg_names.messages] = self.format_messagelist(messages)
            if (
                self.arg_names.response_schema is not None
                and response_schema is not None
            ):
                kwargs[self.arg_names.response_schema] = response_schema
        return {**self.model.call_args, **self.defaults, **kwargs}

    def _call(self, *args, **kwargs):
        """Thread-safe synchronous call with retry logic."""
        return self._sync_call_with_retry(*args, **kwargs)

    async def _acall(self, *args, **kwargs):
        """Thread-safe asynchronous call with retry logic."""
        return await self._async_call_with_retry(*args, **kwargs)


class LiteralCaller(LLMCaller):
    def __init__(self, text: str):
        super().__init__(
            model=LLMModelType(name=None),
            func=lambda: text,
            afunc=self._literalafunc(text),
            token_window=0,
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
