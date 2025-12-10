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

        model_: LLMModelType = Modtype(name=model)

        client = boto3.client(service_name="bedrock-runtime", **model_.client_args)
        aclient = aioboto3.session.Session().client(
            service_name="bedrock-runtime", **model_.client_args
        )

        call_arg_names = LLMCallArgNames(
            model="modelId",
            messages="messages",
            max_tokens=model_.max_token_arg_name,
        )

        super().__init__(
            model=model_,
            func=client.converse,
            afunc=bedrock_async_wrapper(aclient),
            arg_names=call_arg_names,
            defaults={},
            token_window=model_.token_window,
            token_limit_completion=model_.token_limit_completion,
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
        kwargs = super()._proc_call_args(
            messages, max_tokens, response_schema, **kwargs
        )
        inferenceConfig = {}
        inferenceConfig["maxTokens"] = kwargs.pop(self.arg_names.max_tokens)
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
        if message.role == "tool":
            return [
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": tc.id,
                                "content": [
                                    {
                                        "text": tc.response,
                                    }
                                ],
                                "status": "success",
                            }
                        }
                        for tc in message.tool_calls
                    ],
                }
            ]
        ### Handle assistant tool calls messages
        elif message.role == "assistant" and message.tool_calls is not None:
            return {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": tc.id,
                            "name": tc.name,
                            "input": tc.args,
                        }
                    }
                    for tc in message.tool_calls
                ],
            }
        if message.role == "user" and message.images is not None:
            if not self.model.vision:
                raise NotImplementedError(
                    f"Tried to pass images but {self.model.name} doesn't support images"
                )
            for im in message.images:
                if im.url and (im.img is None):
                    raise NotImplementedError(
                        "Bedrock API does not support images by URL"
                    )
            content = [
                {"text": message.message},
                *[
                    {
                        "image": {"format": "png", "source": {"bytes": im.get_bytes()}},
                    }
                    for im in message.images
                ],
            ]
            return {"role": message.role, "content": content}
        return {"role": message.role, "content": [{"text": message.message}]}

    def format_messagelist(self, messagelist: List[LLMMessage]):
        out = []
        sysmsgs = []
        for message in messagelist:
            if message.role == "system":
                sysmsgs.append({"text": message.message})
            else:
                outmsg = self.format_message(message)
                if isinstance(outmsg, list):
                    out.extend(outmsg)
                else:
                    out.append(outmsg)
        if sysmsgs:
            self.defaults["system"] = [sysmsgs[0]]
        else:
            self.defaults.pop("system", None)
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

    def format_output(
        self,
        output: Any,
        response_schema: Optional[BaseModel] = None,
        latency: Optional[float] = None,
    ):
        if isinstance(output, GeneratorType):
            return output
        else:
            content = output["output"]["message"]["content"]
            if output["stopReason"] == "tool_use":
                tool_calls = [c["toolUse"] for c in content if "toolUse" in c]
                tcs = [
                    LLMToolCall(
                        id=tc["toolUseId"],
                        name=tc["name"],
                        args=tc["input"],
                    )
                    for tc in tool_calls
                ]
                return LLMMessage(role="assistant", tool_calls=tcs, latency=latency)
            else:
                text = "".join([c["text"] for c in content if "text" in c]).lstrip()
                # images = [c["image"] for c in content if "image" in c]
                return LLMMessage(role="assistant", message=text, latency=latency)

    def tokenize(self, messagelist: List[LLMMessage]):
        return tokenize_bedrock_messages(self.format_messagelist(messagelist))[0]
