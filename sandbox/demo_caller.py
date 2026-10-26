#%%
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

from fdllm import get_caller
from fdllm.chat import ChatController
from fdllm.llmtypes import LLMMessage
from fdllm.sysutils import register_models, list_models

register_models(Path.home() / ".fdllm/custom_models.yaml")
print(list_models())
#%%
caller_GPT = get_caller("gpt-4.1-mini")
caller_claude = get_caller("claude-sonnet-4-5-20250929")
# caller_mistral = get_caller("fabdata-ai-devel-francecentral-Mistral-large")
caller_gemini = get_caller("gemini-3.5-flash")

#%%
msg = LLMMessage(Role="user", Message="Hi there")

# %%
print(caller_gemini.call(msg))
caller_gemini.response_model_validation_mode = True
print(caller_gemini.call(msg))
#%%
print(caller_GPT.call(msg))

#%%
print(caller_claude.call(msg))

#%%
# print(caller_mistral.call(msg))

#%%
chat = ChatController(Caller=caller_mistral)

#%%
chat.chat("Hi there")

#%%
chat.chat("Can you help me with maths")

#%%
chat.chat("Algebra")