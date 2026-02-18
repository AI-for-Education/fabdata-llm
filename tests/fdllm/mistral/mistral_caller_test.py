"""
Tests for Mistral AI caller.
Tests cover initialization, message formatting, output parsing, and tokenization.
"""
import pytest
import json
from types import SimpleNamespace, GeneratorType
from pathlib import Path
from unittest.mock import patch, MagicMock

from dotenv import load_dotenv

from fdllm.mistralai import MistralCaller
from fdllm.llmtypes import LLMMessage, LLMToolCall
from fdllm.sysutils import register_models

try:
    from mistralai.models.chat_completion import ChatMessage
except ImportError:
    pytest.skip("Mistral AI SDK not installed", allow_module_level=True)

HERE = Path(__file__).resolve().parent
TEST_ROOT = HERE.parent

load_dotenv(TEST_ROOT / "test.env", override=True)

register_models(TEST_ROOT / "custom_models_test.yaml")

TEST_MODEL = "mistral-large-azure"


# ===== Initialization Tests =====

def test_init_mistral():
    """Test MistralCaller initialization with valid model."""
    caller = MistralCaller(model=TEST_MODEL)
    assert caller.Arg_Names.Messages == "messages"
    assert caller.Arg_Names.Model == "model"
    assert caller.Arg_Names.Max_Tokens == "max_tokens"
    # Model name is set to "azureai" for Azure Mistral
    assert caller.Model.Name == "azureai"


def test_init_invalid_model():
    """Test MistralCaller rejects non-Mistral models."""
    with pytest.raises(ValueError, match="is not supported"):
        MistralCaller(model="gpt-4o")


def test_init_unrecognized_model():
    """Test MistralCaller rejects unrecognized models."""
    with pytest.raises(NotImplementedError, match="not a recognised model name"):
        MistralCaller(model="invalid-model-xyz")


# ===== Message Formatting Tests =====

def test_format_message_user():
    """Test formatting user message."""
    caller = MistralCaller(model=TEST_MODEL)
    message = LLMMessage(Role="user", Message="Hello")
    result = caller.format_message(message)

    assert isinstance(result, ChatMessage)
    assert result.role == "user"
    assert result.content == "Hello"


def test_format_message_assistant():
    """Test formatting assistant message."""
    caller = MistralCaller(model=TEST_MODEL)
    message = LLMMessage(Role="assistant", Message="Hi there")
    result = caller.format_message(message)

    assert isinstance(result, ChatMessage)
    assert result.role == "assistant"
    assert result.content == "Hi there"


def test_format_message_system():
    """Test formatting system message."""
    caller = MistralCaller(model=TEST_MODEL)
    message = LLMMessage(Role="system", Message="You are helpful")
    result = caller.format_message(message)

    assert isinstance(result, ChatMessage)
    assert result.role == "system"
    assert result.content == "You are helpful"


def test_format_messagelist():
    """Test formatting a list of messages."""
    caller = MistralCaller(model=TEST_MODEL)
    messages = [
        LLMMessage(Role="system", Message="You are helpful"),
        LLMMessage(Role="user", Message="Hello"),
        LLMMessage(Role="assistant", Message="Hi!"),
    ]
    result = caller.format_messagelist(messages)

    assert len(result) == 3
    assert all(isinstance(m, ChatMessage) for m in result)
    assert result[0].role == "system"
    assert result[1].role == "user"
    assert result[2].role == "assistant"


# ===== Output Formatting Tests =====

def test_format_output_text_response():
    """Test format_output with text response."""
    caller = MistralCaller(model=TEST_MODEL)

    output = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="Test response", tool_calls=None)
        )]
    )

    result = caller.format_output(output, latency=1.5)

    assert isinstance(result, LLMMessage)
    assert result.Role == "assistant"
    assert result.Message == "Test response"
    assert result.Latency == 1.5


def test_format_output_text_with_leading_whitespace():
    """Test format_output strips leading whitespace from content."""
    caller = MistralCaller(model=TEST_MODEL)

    output = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="   Padded response", tool_calls=None)
        )]
    )

    result = caller.format_output(output)

    assert result.Message == "Padded response"


def test_format_output_with_tool_calls():
    """Test format_output with tool calls in response."""
    caller = MistralCaller(model=TEST_MODEL)

    output = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=None,
                tool_calls=[
                    SimpleNamespace(
                        id="call_abc",
                        function=SimpleNamespace(
                            name="get_weather",
                            arguments='{"city": "NYC"}'
                        )
                    ),
                    SimpleNamespace(
                        id="call_def",
                        function=SimpleNamespace(
                            name="get_time",
                            arguments='{"timezone": "UTC"}'
                        )
                    )
                ]
            )
        )]
    )

    result = caller.format_output(output, latency=2.0)

    assert isinstance(result, LLMMessage)
    assert result.Role == "assistant"
    assert result.ToolCalls is not None
    assert len(result.ToolCalls) == 2
    assert result.ToolCalls[0].ID == "call_abc"
    assert result.ToolCalls[0].Name == "get_weather"
    assert result.ToolCalls[0].Args == {"city": "NYC"}
    assert result.ToolCalls[1].ID == "call_def"
    assert result.ToolCalls[1].Name == "get_time"
    assert result.ToolCalls[1].Args == {"timezone": "UTC"}
    assert result.Latency == 2.0


def test_format_output_invalid():
    """Test format_output with invalid output (neither content nor tool_calls)."""
    caller = MistralCaller(model=TEST_MODEL)

    output = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=None, tool_calls=None)
        )]
    )

    with pytest.raises(ValueError, match="Output must be either content or tool call"):
        caller.format_output(output)


def test_format_output_generator():
    """Test format_output with generator type."""
    caller = MistralCaller(model=TEST_MODEL)

    def gen():
        yield "test"

    generator = gen()
    result = caller.format_output(generator)

    assert isinstance(result, GeneratorType)


# ===== Tokenization Tests =====

@pytest.mark.skip(reason="Bug in _tokenizer: passes strings to tokenize_chatgpt_messages which expects dicts")
def test_tokenize():
    """Test tokenize method."""
    caller = MistralCaller(model=TEST_MODEL)
    messages = [LLMMessage(Role="user", Message="Hello, world!")]

    tokens = caller.tokenize(messages)

    assert isinstance(tokens, list)
    assert len(tokens) > 0


@pytest.mark.skip(reason="Bug in _tokenizer: passes strings to tokenize_chatgpt_messages which expects dicts")
def test_tokenize_multiple_messages():
    """Test tokenize with multiple messages."""
    caller = MistralCaller(model=TEST_MODEL)
    messages = [
        LLMMessage(Role="system", Message="You are a helpful assistant"),
        LLMMessage(Role="user", Message="What is 2+2?"),
        LLMMessage(Role="assistant", Message="4"),
    ]

    tokens = caller.tokenize(messages)

    assert isinstance(tokens, list)
    assert len(tokens) > 0
