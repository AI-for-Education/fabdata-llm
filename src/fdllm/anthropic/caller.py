import os
from typing import List
from types import GeneratorType

import anthropic
from anthropic import Anthropic, AsyncAnthropic
from anthropic._tokenizers import sync_get_tokenizer as get_tokenizer

from ..llmtypes import (
    LLMCaller,
    LLMCallArgs,
    AnthropicModelType,
    AnthropicVisionModelType,
    LLMModelType,
    LLMMessage,
)


class ClaudeCaller(LLMCaller):
    def __init__(self, model: str = "claude-2.1"):
        Modtype = LLMModelType.get_type(model)
        if isinstance(Modtype, tuple):
            raise ValueError(f"{model} is ambiguous type")
        if Modtype not in [AnthropicModelType, AnthropicVisionModelType]:
            raise ValueError(f"{model} is not supported")

        model_: LLMModelType = Modtype(Name=model)
        if model_.Client_Args.get("api_key") is None:
            model_.Client_Args["api_key"] = os.environ.get("ANTHROPIC_KEY")

        if Modtype in [AnthropicModelType]:
            client = Anthropic(**model_.Client_Args)
            aclient = AsyncAnthropic(**model_.Client_Args)

        super().__init__(
            Model=model_,
            Func=client.messages.create,
            AFunc=aclient.messages.create,
            Args=LLMCallArgs(
                Model="model", Messages="messages", Max_Tokens="max_tokens"
            ),
            Token_Window=model_.Token_Window,
            Token_Limit_Completion=model_.Token_Limit_Completion,
        )

    def format_message(self, message: LLMMessage):
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

    def format_output(self, output):
        if isinstance(output, GeneratorType):
            return output
        else:
            if output.content is not None:
                return LLMMessage(Role="assistant", Message=output.content[0].text)

    def tokenize(self, messagelist: List[LLMMessage]):
        return tokenizer(self.format_messagelist(messagelist))

def tokenizer(messagelist):
    tokenizer_ = get_tokenizer()
    outstrs = [
        f"role: {msg['role']} content: {msg['content']}" for msg in messagelist
    ]
    return tokenizer_.encode("\n".join(outstrs))