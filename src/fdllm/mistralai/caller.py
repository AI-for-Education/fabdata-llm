from typing import List, Any
from types import GeneratorType
import json
from contextlib import redirect_stdout, redirect_stderr
import os

with open(os.devnull, "w") as null:
    with redirect_stdout(null), redirect_stderr(null):
        from transformers import AutoTokenizer
from mistralai.client import MistralClient
from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import ChatMessage

from ..llmtypes import (
    LLMCaller,
    LLMCallArgs,
    AzureMistralAIModelType,
    LLMModelType,
    LLMMessage,
    LLMToolCall,
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")


class MistralCaller(LLMCaller):
    def __init__(self, model):
        Modtype = LLMModelType.get_type(model)
        if isinstance(Modtype, tuple):
            raise ValueError(f"{model} is ambiguous type")
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
            Args=LLMCallArgs(
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

    def format_output(self, output: Any):
        if isinstance(output, GeneratorType):
            return output
        else:
            return LLMMessage(
                Role="assistant", Message=output.choices[0].message.content.lstrip()
            )

    def tokenize(self, messagelist: List[LLMMessage]):
        return _tokenizer(self.format_messagelist(messagelist))


def _tokenizer(messagelist):
    outstrs = [f"role: {msg.role} content: {msg.content}" for msg in messagelist]
    return tokenizer.encode("\n".join(outstrs))
