import os
import logging
from typing import List, Optional
from types import GeneratorType
from collections import deque
import json
import time

import anthropic
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types.beta import BetaThinkingBlock, BetaToolUseBlock, BetaTextBlock
from anthropic import RateLimitError as RateLimitErrorAnthropic
from ..constants import LLM_DEFAULT_MAX_RETRIES
from pydantic import BaseModel, ConfigDict

from ..llmtypes import (
    LLMCaller,
    LLMCallArgNames,
    AnthropicModelType,
    AnthropicStreamingModelType,
    LLMModelType,
    LLMMessage,
    LLMToolCall,
)
from ..tooluse import Tool
from ..decorators import delayedretry
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_log,
    after_log,
    before_sleep_log,
)


class ClaudeCaller(LLMCaller):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    Client: anthropic._base_client.BaseClient
    AClient: anthropic._base_client.BaseClient

    def __init__(self, model: str = "claude-3-5-sonnet-latest"):
        Modtype = LLMModelType.get_type(model)
        if Modtype not in [AnthropicModelType, AnthropicStreamingModelType]:
            raise ValueError(f"{model} is not supported")

        model_: LLMModelType = Modtype(Name=model)
        client = Anthropic(**model_.Client_Args)
        aclient = AsyncAnthropic(**model_.Client_Args)

        call_arg_names = LLMCallArgNames(
            Model="model",
            Messages="messages",
            Max_Tokens=model_.Max_Token_Arg_Name,
            Response_Schema="tools",
        )

        super().__init__(
            Model=model_,
            Func=client.beta.messages.create,
            AFunc=aclient.beta.messages.create,
            Arg_Names=call_arg_names,
            Token_Window=model_.Token_Window,
            Token_Limit_Completion=model_.Token_Limit_Completion,
            Client=client,
            AClient=aclient,
        )

    def format_message(self, message: LLMMessage):
        if message.Role == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.ID,
                        "content": tc.Response,
                    }
                    for tc in message.ToolCalls
                ],
            }
        elif message.Role == "assistant" and message.ToolCalls is not None:
            out = {"role": "assistant", "content": []}
            if message.Message:
                out["content"].append({"type": "text", "text": message.Message})
            for tc in message.ToolCalls:
                out["content"].append(
                    {
                        "type": "tool_use",
                        "id": tc.ID,
                        "name": tc.Name,
                        "input": tc.Args,
                    }
                )
            return out
        elif message.Role == "user" and message.Images is not None:
            if not self.Model.Vision:
                raise NotImplementedError(
                    f"Tried to pass images but {self.Model.Name} doesn't support images"
                )
            for im in message.Images:
                if im.Url and (im.Img is None):
                    raise NotImplementedError(
                        "Anthropic API does not support images by URL"
                    )
            content = [
                *[
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": im.encode(),
                        },
                    }
                    for im in message.Images
                ],
                {"type": "text", "text": message.Message},
            ]
            return {"role": message.Role, "content": content}
        else:
            return {"role": message.Role, "content": message.Message}

    def format_messagelist(self, messagelist: List[LLMMessage]):
        out = []
        sysmsgs = []
        for message in messagelist:
            if message.Role == "system":
                sysmsgs.append(message.Message)
            else:
                out.append(self.format_message(message))
        if sysmsgs:
            self.Defaults["system"] = sysmsgs[0]
        else:
            self.Defaults.pop("system", None)
        return out

    def format_output(
        self,
        output,
        response_schema: Optional[BaseModel] = None,
        latency: Optional[float] = None,
    ):
        if isinstance(output, GeneratorType):
            return output
        else:
            if getattr(output, "content", None) is not None:
                content = output.content
                if isinstance(content[0], BetaThinkingBlock):
                    thinking = content.pop(0)
                    reasoning_tokens = self.count_tokens(
                        [LLMMessage(Role="assistant", Message=thinking.thinking)]
                    )
                else:
                    reasoning_tokens = 0
                #### token counts
                usage = getattr(output, "usage", None)
                if usage is not None:
                    total_tokens = usage.input_tokens + usage.output_tokens
                    completion_tokens = usage.output_tokens
                else:
                    total_tokens = None
                    completion_tokens = None
                    reasoning_tokens = None
                token_count_kwargs = dict(
                    TokensUsed=total_tokens,
                    TokensUsedCompletion=completion_tokens,
                    TokensUsedReasoning=reasoning_tokens,
                )
                if isinstance(content[0], BetaToolUseBlock):
                    if response_schema is not None:
                        ### if the user has set a response_schema then the tool use block is
                        ### to be processed as an output format, not as a tool call
                        structured_json = output.content[0].input
                        formatted_content = json.dumps(structured_json)
                        out = LLMMessage(
                            Role="assistant",
                            Message=formatted_content,
                            Latency=latency,
                            **token_count_kwargs,
                        )
                    else:
                        # otherwise it should be processed as a tool call
                        out = LLMMessage(
                            Role="assistant",
                            Message="",
                            Latency=latency,
                            **token_count_kwargs,
                        )
                        content = [[], content]
                else:
                    out = LLMMessage(
                        Role="assistant",
                        Message=content[0].text,
                        Latency=latency,
                        **token_count_kwargs,
                    )
                if len(content) > 1:
                    out.ToolCalls = []
                    for tcout in output.content[1:]:
                        tc = LLMToolCall(
                            ID=tcout.id,
                            Name=tcout.name,
                            Args=tcout.input,
                        )
                    out.ToolCalls.append(tc)
            return out

    def format_tool(self, tool: Tool):
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": {
                "type": "object",
                "properties": {key: val.dict() for key, val in tool.params.items()},
                "required": [key for key, val in tool.params.items() if val.required],
            },
        }

    def _proc_call_args(self, messages, max_tokens, response_schema, **kwargs):
        def resolve_refs(indict, refdict={}):
            outdict = indict.copy()
            for key, val in indict.items():
                if isinstance(key, str) and key == "$ref":
                    if len(indict) > 1:
                        raise
                    if isinstance(val, str) and val in refdict:
                        return refdict[val]
                elif isinstance(key, str) and key == "$defs":
                    for refname, refval in val.items():
                        refdict[f"#/$defs/{refname}"] = refval
                elif isinstance(val, dict):
                    outdict[key] = resolve_refs(val, refdict)
                else:
                    outdict[key] = val
            return outdict

        if response_schema is not None:
            response_schema_resolved = resolve_refs(
                response_schema.model_json_schema(), {}
            )
            response_schema = [
                {
                    "name": response_schema_resolved["title"],
                    "input_schema": {
                        "type": "object",
                        "properties": response_schema_resolved["properties"],
                    },
                }
            ]
            kwargs["tool_choice"] = {
                "type": "tool",
                "name": response_schema_resolved["title"],
            }
        kwargs = super()._proc_call_args(
            messages, max_tokens, response_schema, **kwargs
        )
        return kwargs

    # def tokenize(self, messagelist: List[LLMMessage]):
    #     return tokenizer(self.format_messagelist(messagelist))

    def count_tokens(self, messagelist: List[LLMMessage]):
        return self.Client.beta.messages.count_tokens(
            model=self.Model.Api_Model_Name,
            messages=self.format_messagelist(messagelist),
        ).input_tokens


class ClaudeStreamingCaller(ClaudeCaller):
    def __init__(self, model: str = "claude-3-5-sonnet-latest"):
        super().__init__(model)

        self.Func = self.Client.beta.messages.stream
        self.AFunc = self.AClient.beta.messages.stream

        # Override retry methods with streaming-specific implementations
        self._create_streaming_retry_methods()

    def _create_streaming_retry_methods(self):
        """Create streaming-specific Tenacity-based retry methods."""

        # Sync streaming version with Tenacity
        @retry(
            stop=stop_after_attempt(LLM_DEFAULT_MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type((RateLimitErrorAnthropic,)),
            before=before_log(self.logger, logging.DEBUG),
            before_sleep=before_sleep_log(self.logger, logging.WARNING, exc_info=True),
            after=after_log(self.logger, logging.DEBUG),
            reraise=True,
        )
        def sync_streaming_retry(*args, **kwargs):
            start_time = time.perf_counter()
            with self.Func(*args, **kwargs) as stream:
                deque(stream.text_stream, maxlen=0)
                out = stream.get_final_message()
            latency = time.perf_counter() - start_time
            return out, latency

        # Async streaming version with Tenacity
        @retry(
            stop=stop_after_attempt(LLM_DEFAULT_MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type((RateLimitErrorAnthropic,)),
            before=before_log(self.logger, logging.DEBUG),
            before_sleep=before_sleep_log(self.logger, logging.WARNING, exc_info=True),
            after=after_log(self.logger, logging.DEBUG),
            reraise=True,
        )
        async def async_streaming_retry(*args, **kwargs):
            start_time = time.perf_counter()
            async with self.AFunc(*args, **kwargs) as stream:
                async for _ in stream.text_stream:
                    pass
                out = await stream.get_final_message()
            latency = time.perf_counter() - start_time
            return out, latency

        # Override the base class retry methods with streaming versions
        self._sync_call_with_retry = sync_streaming_retry
        self._async_call_with_retry = async_streaming_retry


# def tokenizer(messagelist):
#     tokenizer_ = get_tokenizer()
#     outstrs = [f"role: {msg['role']} content: {msg['content']}" for msg in messagelist]
#     return tokenizer_.encode("\n".join(outstrs))
