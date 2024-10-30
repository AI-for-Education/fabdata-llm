import os
from typing import List
from types import GeneratorType
import json

import anthropic
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import ToolUseBlock
from anthropic._tokenizers import sync_get_tokenizer as get_tokenizer

from ..llmtypes import (
    LLMCaller,
    LLMCallArgs,
    AnthropicModelType,
    LLMModelType,
    LLMMessage,
    LLMToolCall,
)


class ClaudeCaller(LLMCaller):
    def __init__(self, model: str = "claude-3-5-sonnet-latest"):
        Modtype = LLMModelType.get_type(model)
        if Modtype not in [AnthropicModelType]:
            raise ValueError(f"{model} is not supported")
        
        model_: LLMModelType = Modtype(Name=model)
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

    def format_output(self, output):
        if isinstance(output, GeneratorType):
            return output
        else:
            if output.content is not None:
                content = output.content
                if isinstance(content[0], ToolUseBlock):
                    out = LLMMessage(Role="assistant", Message="")
                    output.content = [[], *output.content]
                else:
                    out = LLMMessage(Role="assistant", Message=output.content[0].text)
                if len(output.content) > 1:
                    out.ToolCalls = []
                    for tcout in output.content[1:]:
                        tc = LLMToolCall(
                            ID=tcout.id,
                            Name=tcout.name,
                            Args=tcout.input,
                        )
                    out.ToolCalls.append(tc)
            return out

    def tokenize(self, messagelist: List[LLMMessage]):
        return tokenizer(self.format_messagelist(messagelist))


def tokenizer(messagelist):
    tokenizer_ = get_tokenizer()
    outstrs = [f"role: {msg['role']} content: {msg['content']}" for msg in messagelist]
    return tokenizer_.encode("\n".join(outstrs))
