from typing import ClassVar, List, Any, Optional
from types import GeneratorType
import json
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
from openai.lib._parsing._completions import type_to_response_format_param
from openai.types import Completion
from openai.types.chat import ChatCompletion
from google.auth import default
from google.auth.transport import requests
from pydantic import BaseModel, ConfigDict

from .tokenizer import (
    tokenize_chatgpt_messages,
    tokenize_chatgpt_messages_v2,
    tokenize_completions_messages,
)
from ..sysutils import deepmerge_dicts
from ..batch import OpenAIBatch
from ..constants import LLM_DEFAULT_MAX_TOKENS
from ..llmtypes import (
    LLMCaller,
    LLMCallArgNames,
    OpenAIModelType,
    OpenAICompletionsModelType,
    VertexAIModelType,
    AzureOpenAIModelType,
    LLMModelType,
    LLMBatchRequest,
    BatchSupport,
    LLMMessage,
    LLMToolCall,
    LLMDocument,
)
from ..tooluse import Tool
from ..errors import (
    EmptyLLMResponse,
    InvalidProviderResponse,
    LLMContentFiltered,
    empty_response_error,
    ensure_nonempty,
    provider_error,
    safe_usage,
)


def _sdk_client_args(model: LLMModelType) -> dict:
    # fdllm's retry loop is the only retry layer (see fdllm.retry); an explicit
    # max_retries in Client_Args still wins
    return {"max_retries": 0, **model.Client_Args}


def _raise_if_provider_error(output, error_meta):
    """Raise if an HTTP 200 response carries an error instead of a completion.

    OpenRouter reports failures after it has accepted a request this way: a
    top-level ``error`` object (no choices), or an ``error`` on a choice with
    ``finish_reason: "error"`` next to partial output. Empty error objects,
    which some providers send on success, are ignored (as in LiteLLM).
    """
    choices = getattr(output, "choices", None) or []
    choice = choices[0] if choices else None
    error = getattr(output, "error", None) or getattr(choice, "error", None)
    if isinstance(error, dict):
        code = error.get("code")
        meaningful = bool(error.get("message")) or code is not None
    else:
        code = None
        meaningful = isinstance(error, str) and bool(error)
    if meaningful or getattr(choice, "finish_reason", None) == "error":
        # the provider's message is not attached: it may echo prompt content
        raise provider_error("Provider returned an error", code=code, **error_meta)


# request-option kwargs the SDK accepts but which are not valid in a batch
# request body
_BATCH_REQUEST_OPTION_KEYS = frozenset(
    {"timeout", "extra_headers", "extra_query"}
)


class OpenAICaller(LLMCaller):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    Client: Any
    AClient: Any

    Supports_Batch: ClassVar[bool] = True

    _batch_url: ClassVar[str] = "/v1/chat/completions"
    _batch_output_parser: ClassVar = staticmethod(ChatCompletion.model_validate)

    def __init__(self, model: str):
        Modtype = LLMModelType.get_type(model)
        model_: LLMModelType = Modtype(Name=model)

        client_args = _sdk_client_args(model_)
        if Modtype in [OpenAIModelType, VertexAIModelType]:
            client = OpenAI(**client_args)
            aclient = AsyncOpenAI(**client_args)
        elif Modtype in [AzureOpenAIModelType]:
            client = AzureOpenAI(azure_deployment=model, **client_args)
            aclient = AsyncAzureOpenAI(azure_deployment=model, **client_args)

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
            Client=client,
            AClient=aclient,
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
        ### Handle user messages which contain documents and/or images
        elif message.Role == "user" and (
            message.Documents is not None or message.Images is not None
        ):
            content = []

            # Add documents first
            if message.Documents is not None:
                if not self.Model.Document:
                    raise NotImplementedError(
                        f"Tried to pass documents but {self.Model.Name} doesn't support documents"
                    )
                for doc in message.Documents:
                    content.append(
                        {
                            "type": "file",
                            "file": {
                                "filename": doc.get_filename(),
                                "file_data": f"data:application/pdf;base64,{doc.encode()}",
                            },
                        }
                    )

            # Add images
            if message.Images is not None:
                if not self.Model.Vision:
                    raise NotImplementedError(
                        f"Tried to pass images but {self.Model.Name} doesn't support images"
                    )
                for im in message.Images:
                    content.append(
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
                    )

            # Add text
            if message.Message:
                content.append({"type": "text", "text": message.Message})

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

    def format_output(
        self,
        output: Any,
        response_schema: Optional[type[BaseModel]] = None,
        latency: Optional[float] = None,
    ):
        out = _gpt_common_fmt_output(output, latency, model=self.Model.Name)
        if isinstance(out, LLMMessage):
            out.ResponseModel = self._parse_model_name_string(out.ResponseModel)
        return out

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

    def _batch_request_body(
        self, messages, max_tokens, response_schema, call_kwargs
    ) -> dict:
        kwargs = self._proc_call_args(
            messages, max_tokens, response_schema, **call_kwargs
        )
        body = {
            key: val
            for key, val in kwargs.items()
            if key not in _BATCH_REQUEST_OPTION_KEYS
        }
        body.update(body.pop("extra_body", None) or {})
        return body

    def _batch_jsonl(
        self, requests: list[LLMBatchRequest], max_tokens, response_schema, kwargs
    ) -> bytes:
        return "\n".join(
            json.dumps(
                {
                    "custom_id": req.Custom_Id,
                    "method": "POST",
                    "url": self._batch_url,
                    "body": self._batch_request_body(
                        req.Messages, max_tokens, response_schema, kwargs
                    ),
                }
            )
            for req in requests
        ).encode("utf-8")

    def create_batch(
        self,
        requests,
        max_tokens=LLM_DEFAULT_MAX_TOKENS,
        response_schema=None,
        completion_window: str = "24h",
        **kwargs,
    ) -> OpenAIBatch:
        reqs = self._coerce_batch_requests(requests)
        input_file = self.Client.files.create(
            file=(
                "fdllm-batch.jsonl",
                self._batch_jsonl(reqs, max_tokens, response_schema, kwargs),
                "application/jsonl",
            ),
            purpose="batch",
        )
        batch = self.Client.batches.create(
            input_file_id=input_file.id,
            endpoint=self._batch_url,
            completion_window=completion_window,
        )
        return self.get_batch(batch.id, reqs, response_schema)

    async def acreate_batch(
        self,
        requests,
        max_tokens=LLM_DEFAULT_MAX_TOKENS,
        response_schema=None,
        completion_window: str = "24h",
        **kwargs,
    ) -> OpenAIBatch:
        reqs = self._coerce_batch_requests(requests)
        input_file = await self.AClient.files.create(
            file=(
                "fdllm-batch.jsonl",
                self._batch_jsonl(reqs, max_tokens, response_schema, kwargs),
                "application/jsonl",
            ),
            purpose="batch",
        )
        batch = await self.AClient.batches.create(
            input_file_id=input_file.id,
            endpoint=self._batch_url,
            completion_window=completion_window,
        )
        return self.get_batch(batch.id, reqs, response_schema)

    def get_batch(
        self, batch_id: str, requests, response_schema=None
    ) -> OpenAIBatch:
        return OpenAIBatch(
            self,
            batch_id,
            self._coerce_batch_requests(requests),
            self._batch_output_parser,
            response_schema,
        )

    def _probe_batch_support(self) -> BatchSupport:
        self.Client.batches.list(limit=1)
        return BatchSupport.SUPPORTED

    async def _aprobe_batch_support(self) -> BatchSupport:
        await self.AClient.batches.list(limit=1)
        return BatchSupport.SUPPORTED


def _gpt_common_fmt_output(output, latency, model=None):
    if isinstance(output, GeneratorType):
        return output
    else:
        # get the model used for response
        response_model = getattr(output, "model", None)
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
                TokensUsed=output.usage.total_tokens,
                TokensUsedCompletion=output.usage.completion_tokens,
                TokensUsedReasoning=reasoning_tokens,
            )
        error_meta = dict(
            provider="openai",
            model=model or getattr(output, "model", None),
            response_id=getattr(output, "id", None),
            usage=safe_usage(
                usage, "prompt_tokens", "completion_tokens", "total_tokens"
            ),
        )
        _raise_if_provider_error(output, error_meta)
        if not getattr(output, "choices", None):
            raise EmptyLLMResponse("Empty response: no choices", **error_meta)
        error_meta["stop_reason"] = getattr(output.choices[0], "finish_reason", None)
        msg = output.choices[0].message
        logprobs = getattr(output.choices[0], "logprobs", None)
        if msg.content and msg.content.strip():
            out = LLMMessage(
                Role="assistant",
                Message=msg.content,
                **token_count_kwargs,
                LogProbs=logprobs,
                Latency=latency,
                ResponseModel=response_model,
            )
            return ensure_nonempty(out, **error_meta)
        elif msg.tool_calls:
            tcs = [
                LLMToolCall(
                    ID=tc.id,
                    Name=tc.function.name,
                    Args=json.loads(tc.function.arguments),
                )
                for tc in msg.tool_calls
            ]
            out = LLMMessage(
                Role="assistant",
                ToolCalls=tcs,
                **token_count_kwargs,
                LogProbs=logprobs,
                Latency=latency,
                ResponseModel=response_model,
            )
            return ensure_nonempty(out, **error_meta)
        elif getattr(msg, "refusal", None):
            # the refusal text is model output, so it is not attached to the error
            raise LLMContentFiltered("The model refused to respond", **error_meta)
        else:
            raise empty_response_error(
                "Output must be either content or tool call", **error_meta
            )


class OpenAICompletionsCaller(OpenAICaller):
    """OpenAI Completions API caller that uses the legacy completions endpoint instead of chat completions."""

    _batch_url: ClassVar[str] = "/v1/completions"
    _batch_output_parser: ClassVar = staticmethod(Completion.model_validate)

    def __init__(self, model: str):
        # Get the model type first
        Modtype = LLMModelType.get_type(model)
        if Modtype not in [OpenAICompletionsModelType]:
            raise ValueError(f"{model} is not supported for completions API")

        model_: LLMModelType = Modtype(Name=model)

        # Create clients for completions API
        client_args = _sdk_client_args(model_)
        client = OpenAI(**client_args)
        aclient = AsyncOpenAI(**client_args)

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
            Client=client,
            AClient=aclient,
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
        self,
        output: Any,
        response_schema: Optional[type[BaseModel]] = None,
        latency: Optional[float] = None,
    ) -> LLMMessage:
        """Format the completions API output to LLMMessage format."""
        if isinstance(output, GeneratorType):
            return output
        else:
            # Completions API returns text in choices[0].text instead of choices[0].message.content
            error_meta = dict(
                provider="openai",
                model=self.Model.Name,
                response_id=getattr(output, "id", None),
                usage=safe_usage(
                    getattr(output, "usage", None),
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                ),
            )
            _raise_if_provider_error(output, error_meta)
            choices = getattr(output, "choices", None)
            if not choices:
                raise EmptyLLMResponse("Empty response: no choices", **error_meta)
            choice = choices[0]
            error_meta["stop_reason"] = getattr(choice, "finish_reason", None)
            if not hasattr(choice, "text"):
                raise InvalidProviderResponse(
                    "Unexpected completions API response format", **error_meta
                )
            out = LLMMessage(Role="assistant", Message=choice.text, Latency=latency)
            return ensure_nonempty(out, **error_meta)

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
