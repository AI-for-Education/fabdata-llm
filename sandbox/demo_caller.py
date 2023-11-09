#%%
from fdllm import GPTCaller, ClaudeCaller
from fdllm.chat import ChatController
from fdllm.llmtypes import LLMMessage

#%%
caller_GPT = GPTCaller("gpt-4-0314")
caller_claude = ClaudeCaller("claude-2")

#%%
msg = LLMMessage(Role="user", Message="Hi there")

#%%
print(caller_GPT.call(msg))

#%%
print(caller_claude.call(msg))

#%%
chat = ChatController(Caller=caller_GPT)

#%%
chat.chat("Hi there")

#%%
chat.chat("Can you help me with maths")

#%%
chat.chat("Algebra")