import os
from typing import List, Optional
from types import GeneratorType
import json

import anthropic
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import ToolUseBlock
from anthropic._tokenizers import sync_get_tokenizer as get_tokenizer
from pydantic import BaseModel

from ..llmtypes import (
    LLMCaller,
    LLMCallArgs,
    AnthropicModelType,
    LLMModelType,
    LLMMessage,
    LLMToolCall,
)
from ..tooluse import Tool


class ClaudeCaller(LLMCaller):
    def __init__(self, model: str = "claude-3-5-sonnet-latest"):
        Modtype = LLMModelType.get_type(model)
        if Modtype not in [AnthropicModelType]:
            raise ValueError(f"{model} is not supported")

        model_: LLMModelType = Modtype(Name=model)
        client = Anthropic(**model_.Client_Args)
        aclient = AsyncAnthropic(**model_.Client_Args)

        call_args = LLMCallArgs(
            Model="model",
            Messages="messages",
            Max_Tokens=model_.Max_Token_Arg_Name,
            Response_Schema="tools",
        )

        super().__init__(
            Model=model_,
            Func=client.messages.create,
            AFunc=aclient.messages.create,
            Args=call_args,
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

    def format_output(self, output, response_schema: Optional[BaseModel] = None):
        if isinstance(output, GeneratorType):
            return output
        else:
            if output.content is not None:
                content = output.content
                if isinstance(content[0], ToolUseBlock):
                    if response_schema is not None:
                        ### if the user has set a response_schema then the tool use block is
                        ### to be processed as an output format, not as a tool call
                        structured_json = output.content[0].input
                        formatted_content = json.dumps(structured_json)
                        out = LLMMessage(Role="assistant", Message=formatted_content)
                    else:
                        # otherwise it should be processed as a tool call
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

    def tokenize(self, messagelist: List[LLMMessage]):
        return tokenizer(self.format_messagelist(messagelist))


def tokenizer(messagelist):
    tokenizer_ = get_tokenizer()
    outstrs = [f"role: {msg['role']} content: {msg['content']}" for msg in messagelist]
    return tokenizer_.encode("\n".join(outstrs))
