from typing import List, Any, Optional
from types import GeneratorType
import json

from mistralai.client import MistralClient
from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import ChatMessage
from pydantic import BaseModel

from ..llmtypes import (
    LLMCaller,
    LLMCallArgNames,
    AzureMistralAIModelType,
    LLMModelType,
    LLMMessage,
    LLMToolCall,
)

# NOTE: Removed attempt to download mistral tokenizer from huggingface
# now always uses gpt tokenizer
MISTRALTOKENIZER = False
from ..openai.tokenizer import tokenize_chatgpt_messages
from ..errors import EmptyLLMResponse, empty_response_error, safe_usage

tokenizer = tokenize_chatgpt_messages


class MistralCaller(LLMCaller):
    def __init__(self, model):
        Modtype = LLMModelType.get_type(model)
        if Modtype not in [AzureMistralAIModelType]:
            raise ValueError(f"{model} is not supported")

        model_: LLMModelType = Modtype(Name=model)
        model_.Name = "azureai"

        client = MistralClient(**model_.Client_Args)
        aclient = MistralAsyncClient(**model_.Client_Args)
        super().__init__(
            Model=model_,
            Func=client.chat,
            AFunc=aclient.chat,
            Arg_Names=LLMCallArgNames(
                Model="model",
                Messages="messages",
                Max_Tokens="max_tokens",
            ),
            Defaults={},
            Token_Window=model_.Token_Window,
            Token_Limit_Completion=model_.Token_Limit_Completion,
        )

    def format_message(self, message: LLMMessage):
        return ChatMessage(role=message.Role, content=message.Message)

    def format_messagelist(self, messagelist: List[LLMMessage]):
        return [self.format_message(message) for message in messagelist]

    def format_output(
        self,
        output: Any,
        response_schema: Optional[type[BaseModel]] = None,
        latency: Optional[float] = None,
    ):
        if isinstance(output, GeneratorType):
            return output
        else:
            error_meta = dict(
                provider="mistral",
                model=self.Model.Name,
                response_id=getattr(output, "id", None),
                usage=safe_usage(
                    getattr(output, "usage", None),
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                ),
            )
            if not getattr(output, "choices", None):
                raise EmptyLLMResponse("Empty response: no choices", **error_meta)
            error_meta["stop_reason"] = getattr(output.choices[0], "finish_reason", None)
            msg = output.choices[0].message
            if msg.content:
                return LLMMessage(
                    Role="assistant", Message=msg.content.lstrip(), Latency=latency
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
                return LLMMessage(Role="assistant", ToolCalls=tcs, Latency=latency)
            else:
                raise empty_response_error(
                    "Output must be either content or tool call", **error_meta
                )

    def tokenize(self, messagelist: List[LLMMessage]):
        return _tokenizer(self.format_messagelist(messagelist))


def _tokenizer(messagelist):
    outstrs = [f"role: {msg.role} content: {msg.content}" for msg in messagelist]
    if MISTRALTOKENIZER:
        return tokenizer.encode("\n".join(outstrs))
    else:
        return tokenizer(outstrs)[0]
