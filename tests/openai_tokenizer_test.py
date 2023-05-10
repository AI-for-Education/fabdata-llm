import pytest

from llm.openai.tokenizer import tokenize_chatgpt_messages

def test_tokenize_chatgpt_messages():
    prompt = "I am a some sample question"
    messages = [
        {"role": "user", "content": prompt}
    ]

    result = tokenize_chatgpt_messages(messages)
    assert len(result[0]) == 8

