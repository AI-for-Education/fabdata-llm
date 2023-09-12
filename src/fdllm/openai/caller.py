import os
from typing import List, Any
from types import GeneratorType

import openai

from .tokenizer import tokenize_chatgpt_messages
from ..llmtypes import (
    LLMCaller, LLMCallArgs, ModelTypeLiteral, LLMModelType, LLMMessage
)

class GPTCaller(LLMCaller):
    def __init__(
        self, model: ModelTypeLiteral="gpt-3.5-turbo", api_key=os.environ.get("OPENAI_KEY", "")
    ):
        super().__init__(
            Model = LLMModelType(Name=model),
            Func = openai.ChatCompletion.create,
            AFunc = openai.ChatCompletion.acreate,
            Args = LLMCallArgs(
                Model=(
                    "engine" if model in [
                        "fabdata-openai-devel-gpt4",
                        "fabdata-openai-devel-gpt432k",
                        "fabdata-openai-devel-gpt35",
                        "fabdata-openai-educaid-gpt4",
                    ]
                    else "model"
                ),
                Messages="messages",
                Max_Tokens="max_tokens"
            ),
            APIKey = api_key,
            Defaults = {},
            Token_Window = (
                4096 if model in [
                    "gpt-3.5-turbo",
                    "gpt-3.5-turbo-0301",
                    "gpt-3.5-turbo-0613",
                    "fabdata-openai-devel-gpt35",
                ]
                else 32000 if model == "fabdata-openai-devel-gpt432k"
                else 8000
            ),
        )
    
    def format_message(self, message: LLMMessage):
        return {"role": message.Role, "content": message.Message}
    
    def format_messagelist(self, messagelist: List[LLMMessage]):
        return [self.format_message(message) for message in messagelist]
    
    def format_output(self, output: Any):
        if isinstance(output, GeneratorType):
            return output
        else:
            return LLMMessage(Role="assistant", Message=output.choices[0].message.content)
    
    def tokenize(self, messagelist: List[LLMMessage]):
        return tokenize_chatgpt_messages(
            self.format_messagelist(messagelist)
        )[0]