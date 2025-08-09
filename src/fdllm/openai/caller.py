from typing import List, Any, Optional
from types import GeneratorType
import json
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
import logging
from collections import deque

from openai import (
    OpenAI,
    AsyncOpenAI,
    AzureOpenAI,
    AsyncAzureOpenAI,
    RateLimitError as RateLimitErrorOpenAI,
    APIConnectionError,
)
from openai.lib._parsing._completions import type_to_response_format_param
from openai.types.chat.chat_completion import (
    ChatCompletion,
    Choice,
    ChatCompletionMessage,
)
from google.auth import default
from google.auth.transport import requests
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_log,
    after_log,
    before_sleep_log,
)

from ..constants import LLM_DEFAULT_MAX_RETRIES
from .tokenizer import (
    tokenize_chatgpt_messages,
    tokenize_chatgpt_messages_v2,
    tokenize_completions_messages,
)
from ..sysutils import deepmerge_dicts
from ..llmtypes import (
    LLMCaller,
    LLMCallArgNames,
    OpenAIModelType,
    OpenAIStreamingModelType,
    OpenAICompletionsModelType,
    VertexAIModelType,
    AzureOpenAIModelType,
    LLMModelType,
    LLMMessage,
    LLMToolCall,
)
from ..tooluse import Tool


class OpenAICaller(LLMCaller):
    def __init__(self, model: str = "gpt-3.5-turbo"):
        Modtype = LLMModelType.get_type(model)
        model_: LLMModelType = Modtype(Name=model)

        if Modtype in [OpenAIModelType, OpenAIStreamingModelType, VertexAIModelType]:
            client = OpenAI(**model_.Client_Args)
            aclient = AsyncOpenAI(**model_.Client_Args)
        elif Modtype in [AzureOpenAIModelType]:
            client = AzureOpenAI(azure_deployment=model, **model_.Client_Args)
            aclient = AsyncAzureOpenAI(azure_deployment=model, **model_.Client_Args)

        call_arg_names = LLMCallArgNames(
            Model="model",
            Messages="messages",
            Max_Tokens=model_.Max_Token_Arg_Name,
            Response_Schema="response_format",
        )

        super().__init__(
            Model=model_,
            Func=client.chat.completions.create,
            AFunc=aclient.chat.completions.create,
            Arg_Names=call_arg_names,
            Defaults={},
            Token_Window=model_.Token_Window,
            Token_Limit_Completion=model_.Token_Limit_Completion,
        )

    def format_message(self, message: LLMMessage):
        ### Handle tool results
        if message.Role == "tool":
            return [
                {
                    "role": "tool",
                    "tool_call_id": tc.ID,
                    "name": tc.Name,
                    "content": tc.Response,
                }
                for tc in message.ToolCalls
            ]
        ### Handle assistant tool calls messages
        elif message.Role == "assistant" and message.ToolCalls is not None:
            return {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc.ID,
                        "type": "function",
                        "function": {"arguments": str(tc.Args), "name": tc.Name},
                    }
                    for tc in message.ToolCalls
                ],
            }
        ### Handle user messages which contain images
        elif message.Role == "user" and message.Images is not None:
            if not self.Model.Vision:
                raise NotImplementedError(
                    f"Tried to pass images but {self.Model.Name} doesn't support images"
                )
            content = [
                *[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                im.Url
                                if im.Url is not None
                                else f"data:image/png;base64,{im.encode()}"
                            ),
                            "detail": im.Detail,
                        },
                    }
                    for im in message.Images
                ],
                {"type": "text", "text": message.Message},
            ]
            return {"role": message.Role, "content": content}
        elif message.Role == "system" and (
            self.Model.Name.startswith(("o1-2024", "o3")) or self.Model.Name == "o1"
        ):
            return {"role": "developer", "content": message.Message}
        else:
            return {"role": message.Role, "content": message.Message}

    def format_messagelist(self, messagelist: List[LLMMessage]):
        out = []
        for message in messagelist:
            outmsg = self.format_message(message)
            if isinstance(outmsg, list):
                out.extend(outmsg)
            else:
                out.append(outmsg)
        return out

    def format_output(self, output: Any, response_schema: Optional[BaseModel] = None):
        return _gpt_common_fmt_output(output)

    def tokenize(self, messagelist: List[LLMMessage]):
        if self.Model.Vision:
            texttokens = tokenize_chatgpt_messages_v2(
                self.format_messagelist(messagelist)
            )
            imgtokens = 0
            for msg in messagelist:
                if msg.Images is not None:
                    for img in msg.Images:
                        ntok = img.tokenize()
                        imgtokens += ntok
            return [None] * (texttokens + imgtokens)
        else:
            return tokenize_chatgpt_messages(self.format_messagelist(messagelist))[0]

    def format_tool(self, tool: Tool):
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {key: val.dict() for key, val in tool.params.items()},
                    "required": [
                        key for key, val in tool.params.items() if val.required
                    ],
                },
            },
        }

    def _proc_call_args(self, messages, max_tokens, response_schema, **kwargs):
        if response_schema is not None:
            response_schema = type_to_response_format_param(response_schema)
        kwargs = super()._proc_call_args(
            messages, max_tokens, response_schema, **kwargs
        )
        if "extra_body" in kwargs:
            kwargs["extra_body"] = deepmerge_dicts(
                self.Model.Extra_Body, kwargs["extra_body"]
            )
        else:
            kwargs["extra_body"] = self.Model.Extra_Body
        return kwargs


def _gpt_common_fmt_output(output):
    if isinstance(output, GeneratorType):
        return output
    else:
        token_count_kwargs = dict(
            TokensUsed=output.usage.total_tokens,
            TokensUsedCompletion=output.usage.completion_tokens,
            TokensUsedReasoning=output.usage.completion_tokens_details.reasoning_tokens,
        )
        msg = output.choices[0].message
        if msg.content is not None:
            return LLMMessage(
                Role="assistant",
                Message=msg.content,
                **token_count_kwargs,
            )
        elif msg.tool_calls is not None:
            tcs = [
                LLMToolCall(
                    ID=tc.id,
                    Name=tc.function.name,
                    Args=json.loads(tc.function.arguments),
                )
                for tc in msg.tool_calls
            ]
            return LLMMessage(Role="assistant", ToolCalls=tcs, **token_count_kwargs)
        else:
            raise ValueError("Output must be either content or tool call")


class OpenAICompletionsCaller(OpenAICaller):
    """OpenAI Completions API caller that uses the legacy completions endpoint instead of chat completions."""

    def __init__(self, model: str = "text-davinci-003"):
        # Get the model type first
        Modtype = LLMModelType.get_type(model)
        if Modtype not in [OpenAICompletionsModelType]:
            raise ValueError(f"{model} is not supported for completions API")

        model_: LLMModelType = Modtype(Name=model)

        # Create clients for completions API
        client = OpenAI(**model_.Client_Args)
        aclient = AsyncOpenAI(**model_.Client_Args)

        call_arg_names = LLMCallArgNames(
            Model="model",
            Messages="prompt",  # Key difference: prompt instead of messages
            Max_Tokens=model_.Max_Token_Arg_Name,
            Response_Schema=None,  # Completions API doesn't support structured outputs
        )

        # Initialize LLMCaller directly instead of calling super().__init__
        super(OpenAICaller, self).__init__(
            Model=model_,
            Func=client.completions.create,
            AFunc=aclient.completions.create,
            Arg_Names=call_arg_names,
            Defaults={},
            Token_Window=model_.Token_Window,
            Token_Limit_Completion=model_.Token_Limit_Completion,
        )

    def format_message(self, message: LLMMessage) -> str:
        """Convert a single LLMMessage to a text string for the completions API."""
        if message.Role == "system":
            return f"{message.Message}\n"
        elif message.Role == "user":
            if message.Images is not None:
                raise NotImplementedError(
                    "Images are not supported in the OpenAI completions API. Use the chat completions API instead."
                )
            return f"{message.Message}\n"
        elif message.Role == "assistant":
            if message.ToolCalls is not None:
                # Convert tool calls to text format
                tool_text = ""
                for tc in message.ToolCalls:
                    tool_text += f"{tc.Name}({json.dumps(tc.Args)})"
                    if tc.Response:
                        tool_text += f" -> {tc.Response}"
                    tool_text += "\n"
                if message.Message:
                    return f"{message.Message}\n{tool_text}"
                else:
                    return f"{tool_text}"
            else:
                return f"{message.Message}\n"
        elif message.Role == "tool":
            # Convert tool results to text format
            tool_results = []
            for tc in message.ToolCalls:
                tool_results.append(f"Tool Result ({tc.Name}): {tc.Response}")
            return "\n".join(tool_results) + "\n"
        else:
            return f"{message.Message}\n"

    def format_messagelist(self, messagelist: List[LLMMessage]) -> str:
        """Convert a list of LLMMessages to a single prompt string for the completions API."""
        prompt_parts = []

        if len(messagelist) > 1:
            raise ValueError("OpenAI Completions API only supports one message")

        for message in messagelist:
            formatted_message = self.format_message(message)
            prompt_parts.append(formatted_message)

        # Join all parts and add a final prompt for the assistant to continue
        prompt = "".join(prompt_parts)

        # If the last message wasn't from the assistant, add a prompt for continuation
        # if messagelist and messagelist[-1].Role != "assistant":
        #     prompt += "Assistant: "

        return prompt

    def format_output(
        self, output: Any, response_schema: Optional[BaseModel] = None
    ) -> LLMMessage:
        """Format the completions API output to LLMMessage format."""
        if isinstance(output, GeneratorType):
            return output
        else:
            # Completions API returns text in choices[0].text instead of choices[0].message.content
            if hasattr(output, "choices") and len(output.choices) > 0:
                choice = output.choices[0]
                if hasattr(choice, "text"):
                    return LLMMessage(Role="assistant", Message=choice.text)
                else:
                    raise ValueError("Unexpected completions API response format")
            else:
                raise ValueError("Invalid completions API response")

    def _proc_call_args(self, messages, max_tokens, response_schema, **kwargs):
        """Process call arguments for the completions API."""
        if response_schema is not None:
            raise NotImplementedError(
                "Structured outputs (response_schema) are not supported in the OpenAI completions API. "
                "Use the chat completions API instead."
            )

        # Call parent method but it will use our overridden Arg_Names
        kwargs = super()._proc_call_args(
            messages, max_tokens, response_schema, **kwargs
        )

        # Remove any chat-specific parameters that don't apply to completions
        kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)

        return kwargs

    def tokenize(self, messagelist: List[LLMMessage]):
        return tokenize_completions_messages(self.format_messagelist(messagelist))[0]


class OpenAIStreamingCaller(OpenAICaller):
    def __init__(self, model: str = "gpt-3.5-turbo"):
        super().__init__(model)

        self._stream_kwargs = {
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        self._basefunc = self.Func
        self.Func = self._streaming_func

        self._baseafunc = self.AFunc
        self.AFunc = self._streaming_afunc

        # Override retry methods with streaming-specific implementations
        self._create_streaming_retry_methods()

    def _streaming_func(self, *args, **kwargs):
        _kwargs = {**kwargs, **self._stream_kwargs}
        return self._basefunc(*args, **_kwargs)

    async def _streaming_afunc(self, *args, **kwargs):
        _kwargs = {**kwargs, **self._stream_kwargs}
        return await self._baseafunc(*args, **_kwargs)

    @staticmethod
    def _completion_from_stream(stream):
        response_chunks = list(stream)
        meta_chunk = response_chunks.pop(-1)
        role_chunk = response_chunks.pop(0)
        role = role_chunk.choices[0].delta.role
        assert role == "assistant"
        content = "".join(
            [
                text
                for chunk in response_chunks
                if (text := chunk.choices[0].delta.content) is not None
            ]
        )
        message = ChatCompletionMessage(role=role, content=content)
        finish_reason = response_chunks[-1].choices[0].finish_reason
        choice = Choice(index=0, finish_reason=finish_reason, message=message)
        
        completion_kwargs = meta_chunk.model_dump()
        
        completion_kwargs["object"] = "chat.completion"
        completion_kwargs["choices"] = [choice]
        return ChatCompletion(**completion_kwargs)


    def _create_streaming_retry_methods(self):
        """Create streaming-specific Tenacity-based retry methods."""

        # Sync streaming version with Tenacity
        @retry(
            stop=stop_after_attempt(LLM_DEFAULT_MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type((RateLimitErrorOpenAI,)),
            before=before_log(self.logger, logging.DEBUG),
            before_sleep=before_sleep_log(self.logger, logging.WARNING, exc_info=True),
            after=after_log(self.logger, logging.DEBUG),
            reraise=True,
        )
        def sync_streaming_retry(*args, **kwargs):
            with self.Func(*args, **kwargs) as stream:
                return self._completion_from_stream(stream)

        # Async streaming version with Tenacity
        @retry(
            stop=stop_after_attempt(LLM_DEFAULT_MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type((RateLimitErrorOpenAI,)),
            before=before_log(self.logger, logging.DEBUG),
            before_sleep=before_sleep_log(self.logger, logging.WARNING, exc_info=True),
            after=after_log(self.logger, logging.DEBUG),
            reraise=True,
        )
        async def async_streaming_retry(*args, **kwargs):
            with self.AFunc(*args, **kwargs) as stream:
                return self._completion_from_stream(stream)

        # Override the base class retry methods with streaming versions
        self._sync_call_with_retry = sync_streaming_retry
        self._async_call_with_retry = async_streaming_retry
