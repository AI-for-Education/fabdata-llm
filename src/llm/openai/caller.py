import os
from typing import List, Any

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
                Model="model", Messages="messages", Max_Tokens="max_tokens"
            ),
            APIKey = api_key,
            Defaults = {},
            Token_Window = (
                4096 if model == "gpt-3.5-turbo" else 8000
            ),
        )
    
    def format_message(self, message: LLMMessage):
        return {"role": message.Role, "content": message.Message}
    
    def format_messagelist(self, messagelist: List[LLMMessage]):
        return [self.format_message(message) for message in messagelist]
    
    def format_output(self, output: Any):
        return LLMMessage(Role="assistant", Message=output.choices[0].message.content)
    
    def tokenize(self, messagelist: List[LLMMessage]):
        return tokenize_chatgpt_messages(
            self.format_messagelist(messagelist)
        )[0]