from typing import List, Any, Dict, Optional
from types import GeneratorType
import json

import aioboto3
import asyncio
import boto3
from botocore.exceptions import ClientError
import tiktoken
from pydantic import BaseModel

from ..llmtypes import (
    LLMCaller,
    LLMCallArgNames,
    BedrockModelType,
    LLMModelType,
    LLMMessage,
    LLMToolCall,
)
from ..constants import LLM_DEFAULT_MAX_TOKENS
from ..tooluse import Tool

encoding = tiktoken.get_encoding("gpt2")


def tokenize_bedrock_messages(messages: List[Dict[str, str]]):
    mstr = "\n".join(
        "\n".join(
            ("\n".join([x.get("text", "") for x in m.get("content", [])]), m["role"])
        )
        for m in messages
        if m["role"]
    )
    return encoding.encode(mstr), mstr


def bedrock_async_wrapper(aclient):
    async def converse(*args, **kwargs):
        async with aclient as client:
            return await client.converse(*args, **kwargs)

    return converse


class BedrockCaller(LLMCaller):
    def __init__(self, model: str = ""):
        Modtype = LLMModelType.get_type(model)
        if Modtype not in [BedrockModelType]:
            raise ValueError(f"{model} is not supported")

        model_: LLMModelType = Modtype(Name=model)

        client = boto3.client(service_name="bedrock-runtime", **model_.Client_Args)
        aclient = aioboto3.session.Session().client(
            service_name="bedrock-runtime", **model_.Client_Args
        )

        call_arg_names = LLMCallArgNames(
            Model="modelId",
            Messages="messages",
            Max_Tokens=model_.Max_Token_Arg_Name,
        )

        super().__init__(
            Model=model_,
            Func=client.converse,
            AFunc=bedrock_async_wrapper(aclient),
            Arg_Names=call_arg_names,
            Defaults={},
            Token_Window=model_.Token_Window,
            Token_Limit_Completion=model_.Token_Limit_Completion,
        )

    def __del__(self):
        # Clean up async caller if it has not been used.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.AFunc())
            else:
                loop.run_until_complete(self.AFunc())
        except Exception:
            pass

    def _proc_call_args(self, messages, max_tokens, response_schema, **kwargs):
        # adjust args for bedrock format
        kwargs = super()._proc_call_args(messages, max_tokens, response_schema, **kwargs)
        inferenceConfig = {}
        inferenceConfig["maxTokens"] = kwargs.pop(self.Args.Max_Tokens)
        for arg in ["temperature", "topP", "stopSequences"]:
            if arg in kwargs:
                inferenceConfig[arg] = kwargs.pop(arg)
        kwargs["inferenceConfig"] = inferenceConfig
        if "tools" in kwargs:
            kwargs["toolConfig"] = {}
            kwargs["toolConfig"]["tools"] = kwargs.pop("tools")
        return kwargs

    def format_message(self, message: LLMMessage):
        ### Handle tool results
        if message.Role == "tool":
            return [
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": tc.ID,
                                "content": [
                                    {
                                        "text": tc.Response,
                                    }
                                ],
                                "status": "success",
                            }
                        }
                    for tc in message.ToolCalls ],
                }
            ]
        ### Handle assistant tool calls messages
        elif message.Role == "assistant" and message.ToolCalls is not None:
            return {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": tc.ID,
                            "name": tc.Name,
                            "input": tc.Args,
                        }
                    }
                    for tc in message.ToolCalls
                ],
            }
        if message.Role == "user" and message.Images is not None:
            if not self.Model.Vision:
                raise NotImplementedError(
                    f"Tried to pass images but {self.Model.Name} doesn't support images"
                )
            for im in message.Images:
                if im.Url and (im.Img is None):
                    raise NotImplementedError(
                        "Bedrock API does not support images by URL"
                    )
            content = [
                {"text": message.Message},
                *[
                    {
                        "image": {"format": "png", "source": {"bytes": im.get_bytes()}},
                    }
                    for im in message.Images
                ],
            ]
            return {"role": message.Role, "content": content}
        return {"role": message.Role, "content": [{"text": message.Message}]}

    def format_messagelist(self, messagelist: List[LLMMessage]):
        out = []
        sysmsgs = []
        for message in messagelist:
            if message.Role == "system":
                sysmsgs.append({"text": message.Message})
            else:
                outmsg = self.format_message(message)
                if isinstance(outmsg, list):
                    out.extend(outmsg)
                else:
                    out.append(outmsg)
        if sysmsgs:
            self.Defaults["system"] = [sysmsgs[0]]
        else:
            self.Defaults.pop("system", None)
        return out

    def format_tool(self, tool: Tool):
        return {
            "toolSpec": {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            key: val.dict() for key, val in tool.params.items()
                        },
                        "required": [
                            key for key, val in tool.params.items() if val.required
                        ],
                    },
                },
            }
        }

    def format_output(self, output: Any, response_schema: Optional[BaseModel] = None):
        if isinstance(output, GeneratorType):
            return output
        else:
            content = output["output"]["message"]["content"]
            if output["stopReason"] == "tool_use":
                tool_calls = [c["toolUse"] for c in content if "toolUse" in c]
                tcs = [
                    LLMToolCall(
                        ID=tc["toolUseId"],
                        Name=tc["name"],
                        Args=tc["input"],
                    )
                    for tc in tool_calls
                ]
                return LLMMessage(Role="assistant", ToolCalls=tcs)
            else:
                text = "".join([c["text"] for c in content if "text" in c]).lstrip()
                #images = [c["image"] for c in content if "image" in c]
                return LLMMessage(Role="assistant", Message=text)

    def tokenize(self, messagelist: List[LLMMessage]):
        return tokenize_bedrock_messages(self.format_messagelist(messagelist))[0]
