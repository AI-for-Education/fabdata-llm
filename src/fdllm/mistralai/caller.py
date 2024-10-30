from typing import List, Any
from types import GeneratorType
import json

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

#NOTE: Removed attempt to download mistral tokenizer from huggingface
# now always uses gpt tokenizer
MISTRALTOKENIZER = False
from ..openai.tokenizer import tokenize_chatgpt_messages

tokenizer = tokenize_chatgpt_messages


class MistralCaller(LLMCaller):
    def __init__(self, model):
        Modtype = LLMModelType.get_type(model)
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
            msg = output.choices[0].message
            if msg.content:
                return LLMMessage(Role="assistant", Message=msg.content.lstrip())
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

    def tokenize(self, messagelist: List[LLMMessage]):
        return _tokenizer(self.format_messagelist(messagelist))


def _tokenizer(messagelist):
    outstrs = [f"role: {msg.role} content: {msg.content}" for msg in messagelist]
    if MISTRALTOKENIZER:
        return tokenizer.encode("\n".join(outstrs))
    else:
        return tokenizer(outstrs)[0]
