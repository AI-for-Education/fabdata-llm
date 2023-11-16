import pytest

from fdllm.openai.tokenizer import (
    tokenize_chatgpt_messages,
    tokenize_chatgpt_messages_v2,
)


def test_tokenize_chatgpt_messages():
    prompt = "I am a some sample question"
    messages = [{"role": "user", "content": prompt}]

    result = tokenize_chatgpt_messages(messages)
    assert len(result[0]) == 8

def test_tokenize_chatgpt_messages_v2():
    prompt = "I am a some sample question"
    messages = [{"role": "user", "content": prompt}]

    result = tokenize_chatgpt_messages_v2(messages)
    assert result == 13