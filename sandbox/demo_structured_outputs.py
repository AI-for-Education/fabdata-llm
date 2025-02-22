# %%
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fdllm import get_caller
from fdllm.chat import ChatController
from fdllm.llmtypes import LLMMessage
from pydantic import BaseModel, TypeAdapter
from fdllm.sysutils import register_models, list_models

load_dotenv(override=True)

register_models(Path.home() / ".fdllm/custom_models.yaml")
print(list_models())
# %%
caller_GPT = get_caller("gpt-4o-mini")
caller_claude = get_caller("claude-3-5-haiku-latest")
caller_gemini = get_caller("gemini-2.0-flash")


# %%
class Colors(BaseModel):
    colors: list[Color]
    
class Color(BaseModel):
    name: str
    rgb: list[int, int, int]


msg = LLMMessage(Role="user", Message="List the colors in the rainbow")

# %%
out = caller_gemini.call(
    messages=[msg],
    response_schema=Colors,
)

print(Colors.model_validate_json(out.Message))

#%%
out = caller_GPT.call(
    messages=[msg],
    response_schema=Colors,
)

print(Colors.model_validate_json(out.Message))

#%%
out = caller_claude.call(
    messages=[msg],
    response_schema=Colors,
)

print(Colors.model_validate_json(out.Message))
