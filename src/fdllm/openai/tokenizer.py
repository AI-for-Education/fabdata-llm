from typing import Dict, List
from pathlib import Path

import numpy as np
import tiktoken

encoding = tiktoken.get_encoding("gpt2")

def tokenize_chatgpt_messages(messages: List[Dict[str, str]]):
    mstr = "\n".join(
        "\n".join((m.get("content", ""), m["role"])) for m in messages if m["role"]
    )
    return encoding.encode(mstr), mstr


def tokenize_chatgpt_messages_v2(messages, model="gpt-3.5-turbo-0613"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    # if model == "gpt-3.5-turbo-0613":  # note: future models may deviate from this
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            if isinstance(value, str):
                num_tokens += len(encoding.encode(value))
            elif isinstance(value, list):
                for cont in value:
                    if cont["type"] == "text":
                        num_tokens += len(encoding.encode(cont["text"]))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def tokenize_completions_messages(messages, model="davinci-002"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("gpt2")

    return len(encoding.encode(messages)), None