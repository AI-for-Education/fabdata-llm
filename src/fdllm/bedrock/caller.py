from typing import List, Any, Dict, Optional
from types import GeneratorType
import json

import aioboto3
import boto3
from botocore.exceptions import ClientError
import tiktoken
from itertools import chain

from ..llmtypes import (
    LLMCaller,
    LLMCallArgs,
    BedrockModelType,
    LLMModelType,
    LLMMessage,
    LLMToolCall,
)
from ..constants import LLM_DEFAULT_MAX_TOKENS

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

        call_args = LLMCallArgs(
            Model="modelId",
            Messages="messages",
            Max_Tokens=model_.Max_Token_Arg_Name,
        )

        super().__init__(
            Model=model_,
            Func=client.converse,
            AFunc=bedrock_async_wrapper(aclient),
            Args=call_args,
            Defaults={},
            Token_Window=model_.Token_Window,
            Token_Limit_Completion=model_.Token_Limit_Completion,
        )

    def _proc_call_args(self, messages, max_tokens, **kwargs):
        # adjust args for bedrock format
        kwargs = super()._proc_call_args(messages, max_tokens, **kwargs)
        inferenceConfig = {}
        inferenceConfig["maxTokens"] = kwargs.pop(self.Args.Max_Tokens)
        for arg in ["temperature", "topP", "stopSequences"]:
            if arg in kwargs:
                inferenceConfig[arg] = kwargs.pop(arg)
        kwargs["inferenceConfig"] = inferenceConfig
        return kwargs

    def format_message(self, message: LLMMessage):
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
                        "image": {
                            "format": "png",
                            "source": {
                                "bytes": im.get_bytes()
                            } 
                        },
                    }
                    for im in message.Images
                ],
            ]
            return {"role": message.Role, "content": content}
        return {"role": message.Role, "content": [{"text": message.Message}]}

    def format_messagelist(self, messagelist: List[LLMMessage]):
        out = []
        for message in messagelist:
            outmsg = self.format_message(message)
            if isinstance(outmsg, list):
                out.extend(outmsg)
            else:
                out.append(outmsg)
                
        return out

    def format_output(self, output: Any):
        if isinstance(output, GeneratorType):
            return output
        else:
            msg_text = output["output"]["message"]["content"][0]["text"]
            return LLMMessage(Role="assistant", Message=msg_text.lstrip())

    def tokenize(self, messagelist: List[LLMMessage]):
        return tokenize_bedrock_messages(self.format_messagelist(messagelist))[0]
