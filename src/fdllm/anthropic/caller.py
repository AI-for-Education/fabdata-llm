import os
from typing import List

import anthropic
from anthropic.tokenizer import get_tokenizer

from ..llmtypes import (
    LLMCaller, LLMCallArgs, ModelTypeLiteral, LLMModelType, LLMMessage
)

class ClaudeCaller(LLMCaller):
    def __init__(
        self, model: ModelTypeLiteral="claude-v1", api_key: str=os.environ["ANTHROPIC_KEY"]
    ):
        super().__init__(
            Model = LLMModelType(Name=model),
            Func = anthropic.Client(api_key).completion,
            AFunc = anthropic.Client(api_key).acompletion,
            Args = LLMCallArgs(
                Model="model", Messages="prompt", Max_Tokens="max_tokens_to_sample"
            ),
            APIKey = api_key,
            Defaults = {
                "stop_sequence": [anthropic.HUMAN_PROMPT],
            },
            Token_Window = 100000 if "-100k" in model else 8000
        )
    
    def format_message(self, message: LLMMessage):
        if message.Role in ["user", "system"]:
            return f"{anthropic.HUMAN_PROMPT} {message.Message}{anthropic.AI_PROMPT}"
        elif message.Role == "assistant":
            return f"{anthropic.AI_PROMPT} {message.Message}"
        else:
            return ""
    
    def format_messagelist(self, messagelist: List[LLMMessage]):
        return "".join(self.format_message(message) for message in messagelist)
    
    def format_output(self, output):
        return LLMMessage(Role="assistant", Message=output["completion"][1:])
    
    def tokenize(self, messagelist: List[LLMMessage]):
        tokenizer = get_tokenizer()
        return tokenizer.encode(self.format_messagelist(messagelist))