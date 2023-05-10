from typing import Dict, List
import tiktoken

encoding = tiktoken.get_encoding("gpt2")

def tokenize_chatgpt_messages(messages: List[Dict[str, str]]):
    mstr = "\n".join(
        "\n".join((m["content"], m["role"])) for m in messages if m["role"]
    )
    return encoding.encode(mstr), mstr