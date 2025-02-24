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

from .tokenizer import tokenize_chatgpt_messages, tokenize_chatgpt_messages_v2
from ..sysutils import deepmerge_dicts
from ..llmtypes import (
    LLMCaller,
    LLMCallArgs,
    OpenAIModelType,
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

        if Modtype in [OpenAIModelType, VertexAIModelType]:
            client = OpenAI(**model_.Client_Args)
            aclient = AsyncOpenAI(**model_.Client_Args)
        elif Modtype in [AzureOpenAIModelType]:
            client = AzureOpenAI(azure_deployment=model, **model_.Client_Args)
            aclient = AsyncAzureOpenAI(azure_deployment=model, **model_.Client_Args)

        call_args = LLMCallArgs(
            Model="model",
            Messages="messages",
            Max_Tokens=model_.Max_Token_Arg_Name,
            Response_Schema="response_format",
        )

        super().__init__(
            Model=model_,
            Func=client.chat.completions.create,
            AFunc=aclient.chat.completions.create,
            Args=call_args,
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
        msg = output.choices[0].message
        if msg.content is not None:
            return LLMMessage(Role="assistant", Message=msg.content)
        elif msg.tool_calls is not None:
            tcs = [
                LLMToolCall(
                    ID=tc.id,
                    Name=tc.function.name,
                    Args=json.loads(tc.function.arguments),
                )
                for tc in msg.tool_calls
            ]
            return LLMMessage(Role="assistant", ToolCalls=tcs)
        else:
            raise ValueError("Output must be either content or tool call")
