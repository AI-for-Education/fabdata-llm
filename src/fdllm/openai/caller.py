from typing import List, Any, Optional
from types import GeneratorType
import json
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
from openai.lib._parsing._completions import type_to_response_format_param
from google.auth import default
from google.auth.transport import requests
from pydantic import BaseModel

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
        model_: LLMModelType = Modtype(name=model)

        if Modtype in [OpenAIModelType, VertexAIModelType]:
            client = OpenAI(**model_.client_args)
            aclient = AsyncOpenAI(**model_.client_args)
        elif Modtype in [AzureOpenAIModelType]:
            client = AzureOpenAI(azure_deployment=model, **model_.client_args)
            aclient = AsyncAzureOpenAI(azure_deployment=model, **model_.client_args)

        call_arg_names = LLMCallArgNames(
            model="model",
            messages="messages",
            max_tokens=model_.max_token_arg_name,
            response_schema="response_format",
        )

        super().__init__(
            model=model_,
            func=client.chat.completions.create,
            afunc=aclient.chat.completions.create,
            arg_names=call_arg_names,
            defaults={},
            token_window=model_.token_window,
            token_limit_completion=model_.token_limit_completion,
        )

    def format_message(self, message: LLMMessage):
        ### Handle tool results
        if message.role == "tool":
            return [
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": tc.response,
                }
                for tc in message.tool_calls
            ]
        ### Handle assistant tool calls messages
        elif message.role == "assistant" and message.tool_calls is not None:
            return {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"arguments": str(tc.args), "name": tc.name},
                    }
                    for tc in message.tool_calls
                ],
            }
        ### Handle user messages which contain images
        elif message.role == "user" and message.images is not None:
            if not self.model.vision:
                raise NotImplementedError(
                    f"Tried to pass images but {self.model.name} doesn't support images"
                )
            content = [
                *[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (im.url if im.url is not None else f"data:image/png;base64,{im.encode()}"),
                            "detail": im.detail,
                        },
                    }
                    for im in message.images
                ],
                {"type": "text", "text": message.message},
            ]
            return {"role": message.role, "content": content}
        elif message.role == "system" and (
            self.model.name.startswith(("o1-2024", "o3")) or self.model.name == "o1"
        ):
            return {"role": "developer", "content": message.message}
        else:
            return {"role": message.role, "content": message.message}

    def format_messagelist(self, messagelist: List[LLMMessage]):
        out = []
        for message in messagelist:
            outmsg = self.format_message(message)
            if isinstance(outmsg, list):
                out.extend(outmsg)
            else:
                out.append(outmsg)
        return out

    def format_output(
        self,
        output: Any,
        response_schema: Optional[BaseModel] = None,
        latency: Optional[float] = None,
    ):
        return _gpt_common_fmt_output(output, latency)

    def tokenize(self, messagelist: List[LLMMessage]):
        if self.model.vision:
            texttokens = tokenize_chatgpt_messages_v2(
                self.format_messagelist(messagelist)
            )
            imgtokens = 0
            for msg in messagelist:
                if msg.images is not None:
                    for img in msg.images:
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
                self.model.extra_body, kwargs["extra_body"]
            )
        else:
            kwargs["extra_body"] = self.model.extra_body
        return kwargs


def _gpt_common_fmt_output(output, latency):
    if isinstance(output, GeneratorType):
        return output
    else:
        usage = getattr(output, "usage",  None)
        if usage is None:
            token_count_kwargs = {}
        else:
            ctd = getattr(usage, "completion_tokens_details", None)
            if ctd is None:
                reasoning_tokens = None
            else:
                reasoning_tokens = getattr(ctd, "reasoning_tokens", None)
            token_count_kwargs = dict(
                tokens_used=output.usage.total_tokens,
                tokens_used_completion=output.usage.completion_tokens,
                tokens_used_reasoning=reasoning_tokens,
            )
        msg = output.choices[0].message
        logprobs = getattr(output.choices[0], "logprobs", None)
        if msg.content is not None:
            return LLMMessage(
                role="assistant",
                message=msg.content,
                **token_count_kwargs,
                log_probs=logprobs,
                latency=latency,
            )
        elif msg.tool_calls is not None:
            tcs = [
                LLMToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    args=json.loads(tc.function.arguments),
                )
                for tc in msg.tool_calls
            ]
            return LLMMessage(
                role="assistant",
                tool_calls=tcs,
                **token_count_kwargs,
                log_probs=logprobs,
                latency=latency,
            )
        else:
            raise ValueError("Output must be either content or tool call")


class OpenAICompletionsCaller(OpenAICaller):
    """OpenAI Completions API caller that uses the legacy completions endpoint instead of chat completions."""

    def __init__(self, model: str = "text-davinci-003"):
        # Get the model type first
        Modtype = LLMModelType.get_type(model)
        if Modtype not in [OpenAICompletionsModelType]:
            raise ValueError(f"{model} is not supported for completions API")

        model_: LLMModelType = Modtype(name=model)

        # Create clients for completions API
        client = OpenAI(**model_.client_args)
        aclient = AsyncOpenAI(**model_.client_args)

        call_arg_names = LLMCallArgNames(
            model="model",
            messages="prompt",  # Key difference: prompt instead of messages
            max_tokens=model_.max_token_arg_name,
            response_schema=None,  # Completions API doesn't support structured outputs
        )

        # Initialize LLMCaller directly instead of calling super().__init__
        super(OpenAICaller, self).__init__(
            model=model_,
            func=client.completions.create,
            afunc=aclient.completions.create,
            arg_names=call_arg_names,
            defaults={},
            token_window=model_.token_window,
            token_limit_completion=model_.token_limit_completion,
        )

    def format_message(self, message: LLMMessage) -> str:
        """Convert a single LLMMessage to a text string for the completions API."""
        if message.role == "system":
            return f"{message.message}\n"
        elif message.role == "user":
            if message.images is not None:
                raise NotImplementedError(
                    "Images are not supported in the OpenAI completions API. Use the chat completions API instead."
                )
            return f"{message.message}\n"
        elif message.role == "assistant":
            if message.tool_calls is not None:
                # Convert tool calls to text format
                tool_text = ""
                for tc in message.tool_calls:
                    tool_text += f"{tc.name}({json.dumps(tc.args)})"
                    if tc.response:
                        tool_text += f" -> {tc.response}"
                    tool_text += "\n"
                if message.message:
                    return f"{message.message}\n{tool_text}"
                else:
                    return f"{tool_text}"
            else:
                return f"{message.message}\n"
        elif message.role == "tool":
            # Convert tool results to text format
            tool_results = []
            for tc in message.tool_calls:
                tool_results.append(f"Tool Result ({tc.name}): {tc.response}")
            return "\n".join(tool_results) + "\n"
        else:
            return f"{message.message}\n"

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
        # if messagelist and messagelist[-1].role != "assistant":
        #     prompt += "Assistant: "

        return prompt

    def format_output(
        self,
        output: Any,
        response_schema: Optional[BaseModel] = None,
        latency: Optional[float] = None,
    ) -> LLMMessage:
        """Format the completions API output to LLMMessage format."""
        if isinstance(output, GeneratorType):
            return output
        else:
            # Completions API returns text in choices[0].text instead of choices[0].message.content
            if hasattr(output, "choices") and len(output.choices) > 0:
                choice = output.choices[0]
                if hasattr(choice, "text"):
                    return LLMMessage(
                        Role="assistant", Message=choice.text, Latency=latency
                    )
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
