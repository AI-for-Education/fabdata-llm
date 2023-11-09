import os
from typing import List, Any, Union, Literal
from types import GeneratorType
import base64
from io import BytesIO
from pathlib import Path

from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
from PIL import Image

from .tokenizer import tokenize_chatgpt_messages, tokenize_chatgpt_messages_v2
from ..llmtypes import (
    LLMCaller,
    LLMCallArgs,
    ModelTypeLiteral,
    LLMModelType,
    LLMMessage,
)


class GPTCaller(LLMCaller):
    def __init__(
        self,
        model: ModelTypeLiteral = "gpt-3.5-turbo",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    ):
        super().__init__(
            Model=LLMModelType(Name=model),
            Func=(
                AzureOpenAI(
                    azure_deployment=model,
                ).chat.completions.create
                if _is_azure(model)
                else OpenAI(api_key=api_key).chat.completions.create
            ),
            AFunc=(
                AsyncAzureOpenAI(
                    azure_deployment=model,
                ).chat.completions.create
                if _is_azure(model)
                else AsyncOpenAI(api_key=api_key).chat.completions.create
            ),
            Args=LLMCallArgs(
                Model="model",
                Messages="messages",
                Max_Tokens="max_tokens",
            ),
            APIKey=api_key,
            Defaults={},
            Token_Window=(
                4096
                if model
                in [
                    "gpt-3.5-turbo",
                    "gpt-3.5-turbo-0301",
                    "gpt-3.5-turbo-0613",
                    "fabdata-openai-devel-gpt35",
                    "fabdata-openai-eastus2-gpt35",
                ]
                else 32000
                if model
                in ["fabdata-openai-devel-gpt432k", "fabdata-openai-eastus2-gpt432k"]
                else 128000
                if model == "gpt-4-1106-preview"
                else 8000
            ),
            Token_Limit_Completion=(4096 if model == "gpt-4-1106-preview" else None),
        )

    def format_message(self, message: LLMMessage):
        return {"role": message.Role, "content": message.Message}

    def format_messagelist(self, messagelist: List[LLMMessage]):
        return [self.format_message(message) for message in messagelist]

    def format_output(self, output: Any):
        if isinstance(output, GeneratorType):
            return output
        else:
            return LLMMessage(
                Role="assistant", Message=output.choices[0].message.content
            )

    def tokenize(self, messagelist: List[LLMMessage]):
        return tokenize_chatgpt_messages(self.format_messagelist(messagelist))[0]


class GPTVisionCaller(LLMCaller):
    def __init__(
        self,
        model: ModelTypeLiteral = "gpt-4-vision-preview",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    ):
        super().__init__(
            Model=LLMModelType(Name=model),
            Func=OpenAI(api_key=api_key).chat.completions.create,
            AFunc=AsyncOpenAI(api_key=api_key).chat.completions.create,
            Args=LLMCallArgs(
                Model="model",
                Messages="messages",
                Max_Tokens="max_tokens",
            ),
            APIKey=api_key,
            Defaults={},
            Token_Window=128000,
            Token_Limit_Completion=4096,
        )

    def format_message(self, message: LLMMessage):
        content = [{"type": "text", "text": message.Message}]
        if message.Images is not None:
            content += [
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
            ]
        return {"role": message.Role, "content": content}

    def format_messagelist(self, messagelist: List[LLMMessage]):
        return [self.format_message(message) for message in messagelist]

    def format_output(self, output: Any):
        if isinstance(output, GeneratorType):
            return output
        else:
            return LLMMessage(
                Role="assistant", Message=output.choices[0].message.content
            )

    def tokenize(self, messagelist: List[LLMMessage]):
        texttokens = tokenize_chatgpt_messages_v2(self.format_messagelist(messagelist))
        imgtokens = 0
        for msg in messagelist:
            if msg.Images is not None:
                for img in msg.Images:
                    ntok = img.tokenize()
                    imgtokens += ntok
        return [None] * (texttokens + imgtokens)


def _is_azure(model):
    return model in [
        "fabdata-openai-devel-gpt4",
        "fabdata-openai-devel-gpt432k",
        "fabdata-openai-devel-gpt35",
        "fabdata-openai-eastus2-gpt4",
        "fabdata-openai-eastus2-gpt432k",
        "fabdata-openai-eastus2-gpt35",
        "fabdata-openai-educaid-gpt4",
    ]
