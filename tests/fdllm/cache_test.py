import dotenv
dotenv.load_dotenv()
from typing import List
import pytest

from fdllm.llmtypes import LLMMessage
from fdllm.cache import LLMMessageCache

phone_number = "+4412134567890"
phone_number_for_reset = "+440987654321"

@pytest.fixture()
def empty_cache():
    return LLMMessageCache()

@pytest.fixture(scope="module")
def cache():
    cache = LLMMessageCache()
    cache.add_item(
        phone_number,
        LLMMessage(Role="user", Message="initial message", TokensUsed= 2)
    )
    cache.add_item(
        phone_number_for_reset,
        LLMMessage(Role="user", Message="second message", TokensUsed= 2)
    )
    return cache

def test_get_item_(cache: LLMMessageCache):
    result = cache[phone_number]
    assert len(result) > 0
    assert isinstance(result[0], LLMMessage)

def test_get_user_chat_history(cache: LLMMessageCache):
    result = cache.get_user_chat_history(phone_number)
    assert len(result) > 0
    assert isinstance(result[0], LLMMessage)

def test_get_user_chat_history_empty(empty_cache: LLMMessageCache):
    result = empty_cache.get_user_chat_history(phone_number)
    assert len(result) == 0

def test_get_user_chat_history_last_entry(empty_cache: LLMMessageCache):
    empty_cache.add_item(
        phone_number,
        LLMMessage(Role="user", Message="initial message", TokensUsed= 2)
    )
    empty_cache.add_item(
        phone_number,
        LLMMessage(Role="user", Message="second message", TokensUsed= 2)
    )
    result = empty_cache.get_user_chat_history_last_entry(phone_number)
    assert result is not None
    assert isinstance(result, LLMMessage)
    assert result.Message.lower() == "second message"

def test_get_user_chat_history_last_entry_empty(empty_cache: LLMMessageCache):
    result = empty_cache.get_user_chat_history_last_entry(phone_number)
    assert result is None

def test_add_items(empty_cache: LLMMessageCache):
    final_message = "third message"
    empty_cache.add_item(
        phone_number,
        LLMMessage(Role="user", Message="initial message", TokensUsed= 2)
    )

    initial_add_count = len(empty_cache[phone_number])

    empty_cache.add_items(
        phone_number,
        [
            LLMMessage(Role="user", Message="second message", TokensUsed= 4),
            LLMMessage(Role="user", Message=final_message, TokensUsed= 6)
        ]
    )

    final_cache_count = len(empty_cache[phone_number])

    assert initial_add_count == 1
    assert final_cache_count == 3
    assert empty_cache[phone_number][final_cache_count-1].Message == final_message

def test_reset_cache(cache: LLMMessageCache):
    initial_hist = cache[phone_number_for_reset]

    cache.reset_cache(phone_number_for_reset)
    cleared_hist = cache[phone_number_for_reset]

    assert len(initial_hist) == 1
    assert len(cleared_hist) == 0

def test_add_max_items(empty_cache: LLMMessageCache):
    final_question = "final question"
    final_answer = "final answer"
    history: List[LLMMessage] = []
    Q_and_a_sequence = 0
    for (i, n) in enumerate(range(1, empty_cache.max_user_cache_items + 1)):
        if i % 2 == 0:
            Q_and_a_sequence = Q_and_a_sequence + 1
            history.append(LLMMessage(Role="user", Message=f"question: {Q_and_a_sequence}", TokensUsed=0))
        else:
            history.append(LLMMessage(Role="assistant", Message=f"answer: {Q_and_a_sequence}", TokensUsed=0))


    empty_cache.add_items(
        phone_number,
        history
    )

    initial_add_count = len(empty_cache[phone_number])
    empty_cache.add_items(
        phone_number,
        [
            LLMMessage(Role="user", Message=final_question, TokensUsed=0),
            LLMMessage(Role="assistant", Message=final_answer, TokensUsed=0)
        ]
    )
    final_cache_count = len(empty_cache[phone_number])

    assert initial_add_count == empty_cache.max_user_cache_items
    assert final_cache_count == empty_cache.max_user_cache_items
    assert empty_cache[phone_number][0].Message == "question: 2"
    assert empty_cache[phone_number][final_cache_count-1].Message == final_answer

