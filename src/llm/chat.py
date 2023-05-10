from __future__ import annotations
from typing import (
    List, Tuple, Literal, Optional
)

from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from dataclasses import field

from .constants import (
    LLM_EMPTY_QUESTION,
    LLM_QUESTION_TOO_LONG,
    LLM_INAPPROPRIATE_QUESTION
)
from .prompt_helpers import (
    clean_prompt_string,
    build_prompt_string,
    contains_banned_words
)
from .llmtypes import LLMMessage, LLMCaller, DEFAULT_MAX_TOKENS

@dataclass(config=ConfigDict(validate_assignment=False))
class ChatController:
    Caller: LLMCaller
    History: List[LLMMessage] = field(default_factory=list)
    Sys_Msg: Optional[str] = None
    Context_Type: Tuple[Literal["before", "after"], ...] = ("before",)
    Clean_Prompt: bool = True
    Allow_Banned: bool = False
    Keep_History: bool = True

    def chat(self, prompt: str, max_tokens: int=DEFAULT_MAX_TOKENS, **kwargs):
        new_message, latest_convo = self._prechat(prompt, max_tokens)
        if new_message.Role == "error":
            return new_message, None
        result = self.Caller.call(latest_convo, max_tokens, **kwargs)
        self._postchat(result)
        return new_message, result
    
    async def achat(self, prompt: str, max_tokens: int=DEFAULT_MAX_TOKENS, **kwargs):
        new_message, latest_convo = self._prechat(prompt, max_tokens)
        if new_message.Role == "error":
            return new_message, None
        result = await self.Caller.acall(latest_convo, max_tokens, **kwargs)
        self._postchat(result)
        return new_message, result

    def _prechat(self, prompt, max_tokens):
        final_prompt = self._clean(prompt)
        if final_prompt in [
            LLM_EMPTY_QUESTION,
            LLM_INAPPROPRIATE_QUESTION
        ]:
            return LLMMessage(Role="error", Message=final_prompt), []
        return self._build_latest_convo(final_prompt, max_tokens)

    def _postchat(self, result):
        if self.Keep_History:
            self.History.append(result)

    def _clean(self, prompt):
        if self.Clean_Prompt:
            final_prompt = clean_prompt_string(build_prompt_string(prompt))
        else:
            final_prompt = prompt
        if len(final_prompt) == 0:
            return LLM_EMPTY_QUESTION
        if not self.Allow_Banned and contains_banned_words(final_prompt):
            return LLM_INAPPROPRIATE_QUESTION
        return final_prompt

    def _build_latest_convo(self, prompt, max_tokens):
        def build_messagelist():
            if self.Sys_Msg is not None:
                sys_msg_llmmsg = LLMMessage(Role="system", Message=self.Sys_Msg)
            else:
                sys_msg_llmmsg = None
            if "before" in self.Context_Type and sys_msg_llmmsg is not None:
                latest_convo = [sys_msg_llmmsg, *self.History]
            else:
                latest_convo = self.History
            latest_convo.append(new_message)
            if "after" in self.Context_Type and sys_msg_llmmsg is not None:
                latest_convo.append(sys_msg_llmmsg)
            return latest_convo
        new_message = LLMMessage(Role="user", Message=prompt)
        latest_convo = build_messagelist()
        while (
            len(self.Caller.tokenize(latest_convo))
            > 
            self.Caller.Token_Window - max_tokens
        ):
            if len(self.History) > 0:
                print("Popping history")
                self.History.pop(0)
                latest_convo = build_messagelist()
            else:
                raise LLMError(LLM_QUESTION_TOO_LONG)
        if self.Keep_History:
            self.History.append(new_message)
        return new_message, latest_convo

class LLMError(Exception):
    pass