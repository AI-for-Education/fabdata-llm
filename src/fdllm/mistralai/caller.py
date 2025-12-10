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

tokenizer = tokenize_chatgpt_messages


class MistralCaller(LLMCaller):
    def __init__(self, model):
        Modtype = LLMModelType.get_type(model)
        if Modtype not in [AzureMistralAIModelType]:
            raise ValueError(f"{model} is not supported")

        model_: LLMModelType = Modtype(name=model)
        model_.name = "azureai"

        client = MistralClient(**model_.client_args)
        aclient = MistralAsyncClient(**model_.client_args)
        super().__init__(
            model=model_,
            func=client.chat,
            afunc=aclient.chat,
            arg_names=LLMCallArgNames(
                model="model",
                messages="messages",
                max_tokens="max_tokens",
            ),
            defaults={},
            token_window=model_.token_window,
            token_limit_completion=model_.token_limit_completion,
        )

    def format_message(self, message: LLMMessage):
        return ChatMessage(role=message.role, content=message.message)

    def format_messagelist(self, messagelist: List[LLMMessage]):
        return [self.format_message(message) for message in messagelist]

    def format_output(
        self,
        output: Any,
        response_schema: Optional[BaseModel] = None,
        latency: Optional[float] = None,
    ):
        if isinstance(output, GeneratorType):
            return output
        else:
            msg = output.choices[0].message
            if msg.content:
                return LLMMessage(
                    role="assistant", message=msg.content.lstrip(), latency=latency
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
                return LLMMessage(role="assistant", tool_calls=tcs, latency=latency)
            else:
                raise ValueError("Output must be either content or tool call")

    def tokenize(self, messagelist: List[LLMMessage]):
        return _tokenizer(self.format_messagelist(messagelist))


def _tokenizer(messagelist):
    outstrs = [f"role: {msg.role} content: {msg.content}" for msg in messagelist]
    if MISTRALTOKENIZER:
        return tokenizer.encode("\n".join(outstrs))
    else:
        return tokenizer(outstrs)[0]
