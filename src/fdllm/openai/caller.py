from typing import List, Any
from types import GeneratorType
import json
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
from google.auth import default
from google.auth.transport import requests

from .tokenizer import tokenize_chatgpt_messages, tokenize_chatgpt_messages_v2
from ..llmtypes import (
    LLMCaller,
    LLMCallArgs,
    OpenAIModelType,
    VertexAIModelType,
    AzureOpenAIModelType,
    OpenRouterModelType,
    GroqModelType,
    FireworksModelType,
    LLMModelType,
    LLMMessage,
    LLMToolCall,
)


class GPTCaller(LLMCaller):
    def __init__(self, model: str = "gpt-3.5-turbo"):
        Modtype = LLMModelType.get_type(model)
        if isinstance(Modtype, tuple):
            raise ValueError(f"{model} is ambiguous type")
        if Modtype not in [
            OpenAIModelType,
            AzureOpenAIModelType,
            VertexAIModelType,
            OpenRouterModelType,
            GroqModelType,
            FireworksModelType,
        ]:
            raise ValueError(f"{model} is not supported")

        model_: LLMModelType = Modtype(Name=model)

        if Modtype in [OpenAIModelType]:
            client = OpenAI(**model_.Client_Args)
            aclient = AsyncOpenAI(**model_.Client_Args)
        elif Modtype in [AzureOpenAIModelType]:
            client = AzureOpenAI(azure_deployment=model, **model_.Client_Args)
            aclient = AsyncAzureOpenAI(azure_deployment=model, **model_.Client_Args)
        elif Modtype in [VertexAIModelType]:
            model_.Name = f"google/{model_.Name}"
            if "api_key" not in model_.Client_Args:
                model_.Client_Args["api_key"] = _get_google_token()
            client = OpenAI(**model_.Client_Args)
            aclient = AsyncOpenAI(**model_.Client_Args)
        elif Modtype in [OpenRouterModelType]:
            if model_.Name[:3] != "or-":
                raise ValueError(
                    f"{model._Name} must begin with or- for OpenRouter models"
                )
            if "api_key" not in model_.Client_Args:
                raise ValueError(
                    "api_key must be defined in yaml config for OpenRouter models"
                )
            model_.Name = model_.Model_Prefix + "/" + model_.Name[3:]
            client = OpenAI(**model_.Client_Args)
            aclient = AsyncOpenAI(**model_.Client_Args)
        elif Modtype in [GroqModelType]:
            if model_.Name[:5] != "groq-":
                raise ValueError(f"{model._Name} must begin with groq- for Groq models")
            if "api_key" not in model_.Client_Args:
                raise ValueError(
                    "api_key must be defined in yaml config for Groq models"
                )
            model_.Name = model_.Name[5:]
            client = OpenAI(**model_.Client_Args)
            aclient = AsyncOpenAI(**model_.Client_Args)
        elif Modtype in [FireworksModelType]:
            if model_.Name[:3] != "fw-":
                raise ValueError(
                    f"{model._Name} must begin with fw- for Firework models"
                )
            if "api_key" not in model_.Client_Args:
                raise ValueError(
                    "api_key must be defined in yaml config for Firework models"
                )
            model_.Name = model_.Model_Prefix + "/" + model_.Name[3:]
            client = OpenAI(**model_.Client_Args)
            aclient = AsyncOpenAI(**model_.Client_Args)

        if Modtype in [FireworksModelType, GroqModelType]:
            call_args = LLMCallArgs(
                Model="model",
                Messages="messages",
                Max_Tokens="max_tokens",
            )
        else:
            call_args = LLMCallArgs(
                Model="model",
                Messages="messages",
                Max_Tokens="max_completion_tokens",
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
                {"type": "text", "text": message.Message},
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
            ]
            return {"role": message.Role, "content": content}
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

    def format_output(self, output: Any):
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


class OpenAICaller(LLMCaller):
    def __init__(self, model: str = "gpt-3.5-turbo"):
        Modtype = LLMModelType.get_type(model)
        if isinstance(Modtype, tuple):
            raise ValueError(f"{model} is ambiguous type")
        if Modtype not in [
            OpenAIModelType,
            AzureOpenAIModelType,
            VertexAIModelType,
            OpenRouterModelType,
            GroqModelType,
            FireworksModelType,
        ]:
            raise ValueError(f"{model} is not supported")

        model_: LLMModelType = Modtype(Name=model)

        if Modtype in [OpenAIModelType]:
            client = OpenAI(**model_.Client_Args)
            aclient = AsyncOpenAI(**model_.Client_Args)
        elif Modtype in [AzureOpenAIModelType]:
            client = AzureOpenAI(azure_deployment=model, **model_.Client_Args)
            aclient = AsyncAzureOpenAI(azure_deployment=model, **model_.Client_Args)
        elif Modtype in [VertexAIModelType]:
            model_.Name = f"google/{model_.Name}"
            if "api_key" not in model_.Client_Args:
                model_.Client_Args["api_key"] = _get_google_token()
            client = OpenAI(**model_.Client_Args)
            aclient = AsyncOpenAI(**model_.Client_Args)
        elif Modtype in [OpenRouterModelType]:
            if model_.Name[:3] != "or-":
                raise ValueError(
                    f"{model._Name} must begin with or- for OpenRouter models"
                )
            if "api_key" not in model_.Client_Args:
                raise ValueError(
                    "api_key must be defined in yaml config for OpenRouter models"
                )
            model_.Name = model_.Model_Prefix + "/" + model_.Name[3:]
            client = OpenAI(**model_.Client_Args)
            aclient = AsyncOpenAI(**model_.Client_Args)
        elif Modtype in [GroqModelType]:
            if model_.Name[:5] != "groq-":
                raise ValueError(f"{model._Name} must begin with groq- for Groq models")
            if "api_key" not in model_.Client_Args:
                raise ValueError(
                    "api_key must be defined in yaml config for Groq models"
                )
            model_.Name = model_.Name[5:]
            client = OpenAI(**model_.Client_Args)
            aclient = AsyncOpenAI(**model_.Client_Args)
        elif Modtype in [FireworksModelType]:
            if model_.Name[:3] != "fw-":
                raise ValueError(
                    f"{model._Name} must begin with fw- for Firework models"
                )
            if "api_key" not in model_.Client_Args:
                raise ValueError(
                    "api_key must be defined in yaml config for Firework models"
                )
            model_.Name = model_.Model_Prefix + "/" + model_.Name[3:]
            client = OpenAI(**model_.Client_Args)
            aclient = AsyncOpenAI(**model_.Client_Args)

        if Modtype in [FireworksModelType, GroqModelType]:
            call_args = LLMCallArgs(
                Model="model",
                Messages="messages",
                Max_Tokens="max_tokens",
            )
        else:
            call_args = LLMCallArgs(
                Model="model",
                Messages="messages",
                Max_Tokens="max_completion_tokens",
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
                {"type": "text", "text": message.Message},
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
            ]
            return {"role": message.Role, "content": content}
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

    def format_output(self, output: Any):
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


def _get_google_token():
    def get_token():
        creds, _ = default()
        auth_req = requests.Request()
        creds.refresh(auth_req)
        return creds.token

    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        f = NamedTemporaryFile("w+t", delete=False)
        try:
            cred_json_str = os.environ.get("GOOGLE_AUTH_JSON").replace("\\n", "\n")
            cred_json = json.loads(cred_json_str)
            json.dump(cred_json, f)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name
            f.close()
            token = get_token()
            Path(f.name).unlink()
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS")
        except Exception as e:
            print(e)
            f.close()
            Path(f.name).unlink()
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS")
            token = None
    else:
        token = get_token()
    return token

