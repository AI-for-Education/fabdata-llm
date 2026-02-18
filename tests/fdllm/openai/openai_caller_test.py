"""
Tests for OpenAI caller including chat completions and legacy completions API.
Tests cover basic functionality, tools, images, special models, and metadata handling.
"""
import pytest
import json
from types import SimpleNamespace, GeneratorType
from pathlib import Path
from unittest.mock import patch
from PIL import Image

from dotenv import load_dotenv

from fdllm import OpenAICaller
from fdllm.openai.caller import OpenAICompletionsCaller
from fdllm.llmtypes import LLMMessage, LLMToolCall, LLMImage
from fdllm.openai.tokenizer import tokenize_chatgpt_messages
from fdllm.tooluse import Tool, ToolParam
from fdllm.sysutils import register_models

HERE = Path(__file__).resolve().parent
TEST_ROOT = HERE.parent

load_dotenv(TEST_ROOT / "test.env", override=True)

MESSAGE_ROLES = ("user", "system", "assistant", "error")
TEST_MESSAGE_TEXT = "This is a test"
TEST_MESSAGE = {
    role: LLMMessage(Role=role, Message=("" if role == "error" else TEST_MESSAGE_TEXT))
    for role in MESSAGE_ROLES
}
TEST_MESSAGE_LIST = [TEST_MESSAGE[role] for role in ("system", "user", "assistant")]
TEST_RESULT_OPENAI = SimpleNamespace(
    choices=[SimpleNamespace(
        message=SimpleNamespace(content=TEST_MESSAGE_TEXT)
    )]
)
TEST_MODELS = ["gpt-3.5-turbo", "gpt-4.1-mini"]
TEST_VISION_MODELS = ["gpt-4.1"]

register_models(TEST_ROOT / "custom_models_test.yaml")


# ===== Basic OpenAICaller Tests =====

@pytest.mark.parametrize("model", TEST_MODELS)
def test_init_openai(model):
    OpenAICaller(model=model)


@pytest.mark.parametrize("model", TEST_VISION_MODELS)
def test_init_openaivision(model):
    OpenAICaller(model=model)


@pytest.mark.parametrize(
    "role, expected",
    [
        (role, {"role": role, "content": message.Message})
        for role, message in TEST_MESSAGE.items()
    ],
)
def test_format_message_openai(role, expected):
    caller = OpenAICaller()
    assert caller.format_message(TEST_MESSAGE[role]) == expected


def test_format_messagelist_openai():
    caller = OpenAICaller()
    out = caller.format_messagelist(TEST_MESSAGE_LIST)
    expected = [caller.format_message(message) for message in TEST_MESSAGE_LIST]
    assert out == expected


def test_format_output_openai():
    caller = OpenAICaller()
    out = caller.format_output(TEST_RESULT_OPENAI)
    assert isinstance(out, LLMMessage)


def test_tokenize_openai():
    caller = OpenAICaller()
    out = len(caller.tokenize(TEST_MESSAGE_LIST))
    expected = len(
        tokenize_chatgpt_messages(caller.format_messagelist(TEST_MESSAGE_LIST))[0]
    )
    assert out == expected


# ===== Tool Handling Tests =====

def test_format_message_tool_role():
    """Test formatting of messages with tool role"""
    caller = OpenAICaller()

    tool_calls = [
        LLMToolCall(ID="call_123", Name="get_weather", Args={"city": "NYC"}, Response='{"temp": 72}'),
        LLMToolCall(ID="call_456", Name="get_time", Args={}, Response='{"time": "12:00"}'),
    ]

    message = LLMMessage(Role="tool", ToolCalls=tool_calls)
    result = caller.format_message(message)

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == {
        "role": "tool",
        "tool_call_id": "call_123",
        "name": "get_weather",
        "content": '{"temp": 72}',
    }
    assert result[1] == {
        "role": "tool",
        "tool_call_id": "call_456",
        "name": "get_time",
        "content": '{"time": "12:00"}',
    }


def test_format_message_assistant_tool_calls():
    """Test formatting of assistant messages with tool calls"""
    caller = OpenAICaller()

    tool_calls = [
        LLMToolCall(ID="call_789", Name="search", Args={"query": "test"}),
    ]

    message = LLMMessage(Role="assistant", ToolCalls=tool_calls)
    result = caller.format_message(message)

    assert result == {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_789",
                "type": "function",
                "function": {
                    "arguments": "{'query': 'test'}",
                    "name": "search"
                }
            }
        ]
    }


def test_format_tool():
    """Test format_tool method"""
    caller = OpenAICaller()

    # Create a mock tool
    class TestTool(Tool):
        name = "test_tool"
        description = "A test tool"
        params = {
            "param1": ToolParam(type="string", description="First param", required=True),
            "param2": ToolParam(type="integer", description="Second param", default=42),
        }

        def execute(self, **params):
            return "result"

        async def aexecute(self, **params):
            return "result"

    tool = TestTool()
    result = caller.format_tool(tool)

    assert result == {
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "First param"},
                    "param2": {"type": "integer", "description": "Second param"},
                },
                "required": ["param1"],
            },
        },
    }


def test_format_messagelist_with_tool_messages():
    """Test format_messagelist with tool messages that get extended"""
    caller = OpenAICaller()

    tool_calls = [
        LLMToolCall(ID="call_1", Name="tool1", Response="result1"),
        LLMToolCall(ID="call_2", Name="tool2", Response="result2"),
    ]

    messages = [
        LLMMessage(Role="user", Message="test"),
        LLMMessage(Role="tool", ToolCalls=tool_calls),
    ]

    result = caller.format_messagelist(messages)

    # Should have 3 messages total: 1 user + 2 tool results
    assert len(result) == 3
    assert result[0] == {"role": "user", "content": "test"}
    assert result[1]["role"] == "tool"
    assert result[2]["role"] == "tool"


# ===== Image Handling Tests =====

def test_format_message_with_images():
    """Test formatting of user messages with images"""
    caller = OpenAICaller(model="gpt-4.1")  # Vision model

    # Create a simple test image
    img = Image.new('RGB', (10, 10), color='red')
    llm_image = LLMImage(Img=img, Detail="high")

    message = LLMMessage(Role="user", Message="What's in this image?", Images=[llm_image])
    result = caller.format_message(message)

    assert result["role"] == "user"
    assert isinstance(result["content"], list)
    assert len(result["content"]) == 2  # image + text
    assert result["content"][0]["type"] == "image_url"
    assert result["content"][0]["image_url"]["detail"] == "high"
    assert result["content"][1] == {"type": "text", "text": "What's in this image?"}


def test_format_message_with_image_url():
    """Test formatting with image URL"""
    caller = OpenAICaller(model="gpt-4.1")  # Vision model

    llm_image = LLMImage(Url="https://example.com/image.png", Detail="low")
    message = LLMMessage(Role="user", Message="Analyze", Images=[llm_image])
    result = caller.format_message(message)

    assert result["content"][0]["image_url"]["url"] == "https://example.com/image.png"
    assert result["content"][0]["image_url"]["detail"] == "low"


def test_format_message_images_non_vision_model():
    """Test that images raise error for non-vision models"""
    caller = OpenAICaller(model="gpt-3.5-turbo")  # Non-vision model

    img = Image.new('RGB', (10, 10))
    llm_image = LLMImage(Img=img)
    message = LLMMessage(Role="user", Message="test", Images=[llm_image])

    with pytest.raises(NotImplementedError, match="doesn't support images"):
        caller.format_message(message)


def test_tokenize_with_vision():
    """Test tokenization for vision models with images"""
    caller = OpenAICaller(model="gpt-4.1")  # Vision model

    img = Image.new('RGB', (100, 100))
    llm_image = LLMImage(Img=img, Detail="low")

    messages = [
        LLMMessage(Role="user", Message="What's this?", Images=[llm_image])
    ]

    tokens = caller.tokenize(messages)

    # Should return a list (None tokens for text + image tokens)
    assert isinstance(tokens, list)
    # Should have tokens for both text and image
    assert len(tokens) > 0


# ===== Special Model Handling Tests =====

@pytest.mark.parametrize("model_name", [
    "o1-2024",
    "o1",
    "o3-mini",
])
def test_format_message_reasoning_models_system_to_developer(model_name):
    """Test o1/o3 reasoning models convert system role to developer"""
    caller = OpenAICaller(model="gpt-3.5-turbo")
    caller.Model.Name = model_name

    message = LLMMessage(Role="system", Message="You are a helpful assistant")
    result = caller.format_message(message)

    assert result["role"] == "developer"
    assert result["content"] == "You are a helpful assistant"


# ===== Token Usage & Metadata Tests =====

def test_format_output_with_usage_and_reasoning_tokens():
    """Test format_output with reasoning tokens"""
    caller = OpenAICaller()

    # Mock output with reasoning tokens
    output = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="Test response"),
            logprobs={"token_logprobs": [0.1, 0.2]}
        )],
        usage=SimpleNamespace(
            total_tokens=100,
            completion_tokens=50,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=20)
        )
    )

    result = caller.format_output(output, latency=1.5)

    assert isinstance(result, LLMMessage)
    assert result.Message == "Test response"
    assert result.TokensUsed == 100
    assert result.TokensUsedCompletion == 50
    assert result.TokensUsedReasoning == 20
    assert result.LogProbs is not None
    assert result.Latency == 1.5


def test_format_output_without_reasoning_tokens():
    """Test format_output without completion_tokens_details"""
    caller = OpenAICaller()

    output = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="Test"),
            logprobs=None
        )],
        usage=SimpleNamespace(
            total_tokens=50,
            completion_tokens=25,
            # No completion_tokens_details
        )
    )

    result = caller.format_output(output)

    assert result.TokensUsedReasoning is None


def test_format_output_with_tool_calls():
    """Test format_output with tool calls in response"""
    caller = OpenAICaller()

    output = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=None,
                tool_calls=[
                    SimpleNamespace(
                        id="call_abc",
                        function=SimpleNamespace(
                            name="get_data",
                            arguments='{"param": "value"}'
                        )
                    )
                ]
            ),
            logprobs=None
        )],
        usage=SimpleNamespace(
            total_tokens=75,
            completion_tokens=25,
            completion_tokens_details=None
        )
    )

    result = caller.format_output(output, latency=2.0)

    assert isinstance(result, LLMMessage)
    assert result.Role == "assistant"
    assert result.ToolCalls is not None
    assert len(result.ToolCalls) == 1
    assert result.ToolCalls[0].ID == "call_abc"
    assert result.ToolCalls[0].Name == "get_data"
    assert result.ToolCalls[0].Args == {"param": "value"}
    assert result.Latency == 2.0


def test_format_output_invalid():
    """Test format_output with invalid output"""
    caller = OpenAICaller()

    # Output with neither content nor tool_calls
    output = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=None, tool_calls=None),
            logprobs=None
        )],
        usage=SimpleNamespace(total_tokens=10, completion_tokens=5)
    )

    with pytest.raises(ValueError, match="Output must be either content or tool call"):
        caller.format_output(output)


def test_format_output_generator():
    """Test format_output with generator type"""
    caller = OpenAICaller()

    def gen():
        yield "test"

    generator = gen()
    result = caller.format_output(generator)

    assert isinstance(result, GeneratorType)


# ===== _proc_call_args Tests =====

def test_proc_call_args_with_response_schema():
    """Test _proc_call_args with response_schema"""
    from pydantic import BaseModel

    class ResponseSchema(BaseModel):
        result: str

    caller = OpenAICaller()
    messages = [LLMMessage(Role="user", Message="test")]

    # Mock the type_to_response_format_param function
    with patch('fdllm.openai.caller.type_to_response_format_param') as mock_convert:
        mock_convert.return_value = {"type": "json_schema", "json_schema": {}}

        kwargs = caller._proc_call_args(messages, 100, ResponseSchema)

        mock_convert.assert_called_once_with(ResponseSchema)
        assert "response_format" in kwargs


def test_proc_call_args_extra_body_merge():
    """Test _proc_call_args merging extra_body"""
    caller = OpenAICaller()
    caller.Model.Extra_Body = {"default_key": "default_value"}

    messages = [LLMMessage(Role="user", Message="test")]

    kwargs = caller._proc_call_args(
        messages, 100, None,
        extra_body={"custom_key": "custom_value"}
    )

    assert "extra_body" in kwargs
    assert kwargs["extra_body"]["default_key"] == "default_value"
    assert kwargs["extra_body"]["custom_key"] == "custom_value"


def test_proc_call_args_extra_body_default():
    """Test _proc_call_args with default extra_body"""
    caller = OpenAICaller()
    caller.Model.Extra_Body = {"key": "value"}

    messages = [LLMMessage(Role="user", Message="test")]

    kwargs = caller._proc_call_args(messages, 100, None)

    assert kwargs["extra_body"] == {"key": "value"}


# ===== OpenAICompletionsCaller Tests =====

def test_completions_caller_init():
    """Test OpenAICompletionsCaller initialization"""
    # This will test with a mock completions model
    with patch('fdllm.openai.caller.OpenAI'):
        with patch('fdllm.openai.caller.AsyncOpenAI'):
            # We need to register a completions model
            caller = OpenAICompletionsCaller(model="text-davinci-003")

            assert caller.Arg_Names.Messages == "prompt"
            assert caller.Arg_Names.Response_Schema is None


def test_completions_caller_invalid_model():
    """Test OpenAICompletionsCaller with non-completions model"""
    with pytest.raises(ValueError, match="not supported for completions API"):
        OpenAICompletionsCaller(model="gpt-3.5-turbo")


def test_completions_format_message_system():
    """Test completions format_message for system role"""
    caller = OpenAICompletionsCaller(model="text-davinci-003")

    message = LLMMessage(Role="system", Message="You are helpful")
    result = caller.format_message(message)

    assert result == "You are helpful\n"


def test_completions_format_message_user():
    """Test completions format_message for user role"""
    caller = OpenAICompletionsCaller(model="text-davinci-003")

    message = LLMMessage(Role="user", Message="Hello")
    result = caller.format_message(message)

    assert result == "Hello\n"


def test_completions_format_message_user_with_images():
    """Test completions format_message rejects images"""
    caller = OpenAICompletionsCaller(model="text-davinci-003")

    img = Image.new('RGB', (10, 10))
    llm_image = LLMImage(Img=img)
    message = LLMMessage(Role="user", Message="test", Images=[llm_image])

    with pytest.raises(NotImplementedError, match="Images are not supported"):
        caller.format_message(message)


def test_completions_format_message_assistant_with_tool_calls():
    """Test completions format_message with tool calls"""
    caller = OpenAICompletionsCaller(model="text-davinci-003")

    tool_calls = [
        LLMToolCall(ID="1", Name="tool1", Args={"x": 1}, Response="result1"),
        LLMToolCall(ID="2", Name="tool2", Args={"y": 2}),
    ]

    message = LLMMessage(Role="assistant", ToolCalls=tool_calls, Message="Calling tools")
    result = caller.format_message(message)

    assert "Calling tools" in result
    assert 'tool1({"x": 1}) -> result1' in result
    assert 'tool2({"y": 2})' in result


def test_completions_format_message_assistant_tool_calls_no_message():
    """Test completions format_message with tool calls but no message"""
    caller = OpenAICompletionsCaller(model="text-davinci-003")

    tool_calls = [LLMToolCall(ID="1", Name="tool", Args={})]
    message = LLMMessage(Role="assistant", ToolCalls=tool_calls)

    result = caller.format_message(message)

    assert result == 'tool({})\n'


def test_completions_format_message_assistant_no_tool_calls():
    """Test completions format_message for assistant without tool calls"""
    caller = OpenAICompletionsCaller(model="text-davinci-003")

    message = LLMMessage(Role="assistant", Message="This is a response")
    result = caller.format_message(message)

    assert result == "This is a response\n"


def test_completions_format_message_other_role():
    """Test completions format_message for other roles"""
    caller = OpenAICompletionsCaller(model="text-davinci-003")

    # Test with an error role or any other role
    message = LLMMessage(Role="error", Message="An error occurred")
    result = caller.format_message(message)

    assert result == "An error occurred\n"


def test_completions_format_message_tool_role():
    """Test completions format_message for tool role"""
    caller = OpenAICompletionsCaller(model="text-davinci-003")

    tool_calls = [
        LLMToolCall(ID="1", Name="tool1", Response="result1"),
        LLMToolCall(ID="2", Name="tool2", Response="result2"),
    ]

    message = LLMMessage(Role="tool", ToolCalls=tool_calls)
    result = caller.format_message(message)

    assert "Tool Result (tool1): result1" in result
    assert "Tool Result (tool2): result2" in result


def test_completions_format_messagelist():
    """Test completions format_messagelist"""
    caller = OpenAICompletionsCaller(model="text-davinci-003")

    messages = [LLMMessage(Role="user", Message="Hello")]
    result = caller.format_messagelist(messages)

    assert result == "Hello\n"


def test_completions_format_messagelist_multiple_messages():
    """Test completions format_messagelist rejects multiple messages"""
    caller = OpenAICompletionsCaller(model="text-davinci-003")

    messages = [
        LLMMessage(Role="user", Message="First"),
        LLMMessage(Role="assistant", Message="Second"),
    ]

    with pytest.raises(ValueError, match="only supports one message"):
        caller.format_messagelist(messages)


def test_completions_format_output():
    """Test completions format_output"""
    caller = OpenAICompletionsCaller(model="text-davinci-003")

    output = SimpleNamespace(
        choices=[SimpleNamespace(text="Completion result")]
    )

    result = caller.format_output(output, latency=1.0)

    assert isinstance(result, LLMMessage)
    assert result.Role == "assistant"
    assert result.Message == "Completion result"
    assert result.Latency == 1.0


def test_completions_format_output_invalid():
    """Test completions format_output with invalid response"""
    caller = OpenAICompletionsCaller(model="text-davinci-003")

    # Missing choices
    output = SimpleNamespace(choices=[])

    with pytest.raises(ValueError, match="Invalid completions API response"):
        caller.format_output(output)


def test_completions_format_output_no_text():
    """Test completions format_output without text attribute"""
    caller = OpenAICompletionsCaller(model="text-davinci-003")

    output = SimpleNamespace(
        choices=[SimpleNamespace(message="wrong")]
    )

    with pytest.raises(ValueError, match="Unexpected completions API response format"):
        caller.format_output(output)


def test_completions_format_output_generator():
    """Test completions format_output with generator"""
    caller = OpenAICompletionsCaller(model="text-davinci-003")

    def gen():
        yield "test"

    generator = gen()
    result = caller.format_output(generator)

    assert isinstance(result, GeneratorType)


def test_completions_proc_call_args_response_schema_error():
    """Test completions _proc_call_args rejects response_schema"""
    from pydantic import BaseModel

    class Schema(BaseModel):
        field: str

    caller = OpenAICompletionsCaller(model="text-davinci-003")
    messages = [LLMMessage(Role="user", Message="test")]

    with pytest.raises(NotImplementedError, match="Structured outputs.*not supported"):
        caller._proc_call_args(messages, 100, Schema)


def test_completions_proc_call_args_removes_tools():
    """Test completions _proc_call_args removes tool parameters"""
    caller = OpenAICompletionsCaller(model="text-davinci-003")
    messages = [LLMMessage(Role="user", Message="test")]

    kwargs = caller._proc_call_args(
        messages, 100, None,
        tools=["tool1"],
        tool_choice="auto"
    )

    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs


def test_completions_tokenize():
    """Test completions tokenize method"""
    caller = OpenAICompletionsCaller(model="text-davinci-003")
    messages = [LLMMessage(Role="user", Message="Hello")]

    tokens = caller.tokenize(messages)

    assert isinstance(tokens, list)
    assert len(tokens) > 0
