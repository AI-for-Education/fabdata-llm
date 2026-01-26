"""
Tests for Amazon Bedrock caller.
Tests cover initialization, message formatting, tools, images, system messages, and output parsing.
"""
import pytest
from types import SimpleNamespace, GeneratorType
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

from dotenv import load_dotenv

from fdllm.bedrock import BedrockCaller
from fdllm.bedrock.caller import tokenize_bedrock_messages, bedrock_async_wrapper
from fdllm.llmtypes import LLMMessage, LLMToolCall, LLMImage
from fdllm.tooluse import Tool, ToolParam
from fdllm.sysutils import register_models

HERE = Path(__file__).resolve().parent
TEST_ROOT = HERE.parent

load_dotenv(TEST_ROOT / "test.env", override=True)

register_models(TEST_ROOT / "custom_models_test.yaml")

TEST_MODEL = "bedrock-claude-sonnet"
TEST_VISION_MODEL = "bedrock-claude-sonnet-vision"


# ===== Helper Classes =====

class SampleTool(Tool):
    """Reusable test tool with standard params."""
    name = "sample_tool"
    description = "A sample tool for testing"
    params = {
        "required_param": ToolParam(type="string", description="Required", required=True),
        "optional_param": ToolParam(type="integer", description="Optional", default=0),
    }

    def execute(self, **params):
        return "result"

    async def aexecute(self, **params):
        return "result"


# ===== Initialization Tests =====

@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_init_bedrock(mock_aioboto3, mock_boto3):
    """Test BedrockCaller initialization with valid model."""
    mock_boto3.client.return_value = MagicMock()
    mock_aioboto3.session.Session.return_value.client.return_value = MagicMock()

    caller = BedrockCaller(model=TEST_MODEL)

    assert caller.Arg_Names.Messages == "messages"
    assert caller.Arg_Names.Model == "modelId"
    mock_boto3.client.assert_called_once()


@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_init_invalid_model(mock_aioboto3, mock_boto3):
    """Test BedrockCaller rejects non-Bedrock models."""
    with pytest.raises(ValueError, match="is not supported"):
        BedrockCaller(model="gpt-4o")


@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_init_unrecognized_model(mock_aioboto3, mock_boto3):
    """Test BedrockCaller rejects unrecognized models."""
    with pytest.raises(NotImplementedError, match="not a recognised model name"):
        BedrockCaller(model="invalid-model-xyz")


# ===== Message Formatting Tests =====

@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_format_message_user(mock_aioboto3, mock_boto3):
    """Test formatting user message."""
    mock_boto3.client.return_value = MagicMock()
    mock_aioboto3.session.Session.return_value.client.return_value = MagicMock()

    caller = BedrockCaller(model=TEST_MODEL)
    message = LLMMessage(Role="user", Message="Hello")
    result = caller.format_message(message)

    assert result == {"role": "user", "content": [{"text": "Hello"}]}


@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_format_message_assistant(mock_aioboto3, mock_boto3):
    """Test formatting assistant message."""
    mock_boto3.client.return_value = MagicMock()
    mock_aioboto3.session.Session.return_value.client.return_value = MagicMock()

    caller = BedrockCaller(model=TEST_MODEL)
    message = LLMMessage(Role="assistant", Message="Hi there")
    result = caller.format_message(message)

    assert result == {"role": "assistant", "content": [{"text": "Hi there"}]}


@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_format_message_tool_results(mock_aioboto3, mock_boto3):
    """Test formatting tool results (tool role -> user role with toolResult)."""
    mock_boto3.client.return_value = MagicMock()
    mock_aioboto3.session.Session.return_value.client.return_value = MagicMock()

    caller = BedrockCaller(model=TEST_MODEL)
    tool_calls = [
        LLMToolCall(ID="call_1", Name="tool1", Args={}, Response='{"result": 1}'),
        LLMToolCall(ID="call_2", Name="tool2", Args={}, Response='{"result": 2}'),
    ]
    message = LLMMessage(Role="tool", ToolCalls=tool_calls)
    result = caller.format_message(message)

    # Tool results are wrapped in a list for Bedrock
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert len(result[0]["content"]) == 2
    assert result[0]["content"][0]["toolResult"]["toolUseId"] == "call_1"
    assert result[0]["content"][0]["toolResult"]["content"][0]["text"] == '{"result": 1}'
    assert result[0]["content"][0]["toolResult"]["status"] == "success"
    assert result[0]["content"][1]["toolResult"]["toolUseId"] == "call_2"


@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_format_message_assistant_tool_calls(mock_aioboto3, mock_boto3):
    """Test formatting assistant messages with tool calls."""
    mock_boto3.client.return_value = MagicMock()
    mock_aioboto3.session.Session.return_value.client.return_value = MagicMock()

    caller = BedrockCaller(model=TEST_MODEL)
    tool_calls = [
        LLMToolCall(ID="call_1", Name="search", Args={"query": "test"}),
        LLMToolCall(ID="call_2", Name="lookup", Args={"id": 123}),
    ]
    message = LLMMessage(Role="assistant", ToolCalls=tool_calls)
    result = caller.format_message(message)

    assert result["role"] == "assistant"
    assert len(result["content"]) == 2
    assert result["content"][0]["toolUse"]["toolUseId"] == "call_1"
    assert result["content"][0]["toolUse"]["name"] == "search"
    assert result["content"][0]["toolUse"]["input"] == {"query": "test"}
    assert result["content"][1]["toolUse"]["toolUseId"] == "call_2"
    assert result["content"][1]["toolUse"]["name"] == "lookup"


# ===== Image Handling Tests =====

@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_format_message_with_images(mock_aioboto3, mock_boto3):
    """Test formatting user message with images."""
    mock_boto3.client.return_value = MagicMock()
    mock_aioboto3.session.Session.return_value.client.return_value = MagicMock()

    caller = BedrockCaller(model=TEST_VISION_MODEL)

    img = Image.new('RGB', (10, 10), color='red')
    llm_image = LLMImage(Img=img)

    message = LLMMessage(Role="user", Message="What's in this image?", Images=[llm_image])
    result = caller.format_message(message)

    assert result["role"] == "user"
    assert len(result["content"]) == 2
    assert result["content"][0] == {"text": "What's in this image?"}
    assert "image" in result["content"][1]
    assert result["content"][1]["image"]["format"] == "png"
    assert "bytes" in result["content"][1]["image"]["source"]


@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_format_message_images_non_vision_model(mock_aioboto3, mock_boto3):
    """Test that images raise error for non-vision models."""
    mock_boto3.client.return_value = MagicMock()
    mock_aioboto3.session.Session.return_value.client.return_value = MagicMock()

    caller = BedrockCaller(model=TEST_MODEL)

    img = Image.new('RGB', (10, 10))
    llm_image = LLMImage(Img=img)
    message = LLMMessage(Role="user", Message="test", Images=[llm_image])

    with pytest.raises(NotImplementedError, match="doesn't support images"):
        caller.format_message(message)


@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_format_message_images_url_not_supported(mock_aioboto3, mock_boto3):
    """Test that image URLs raise error for Bedrock."""
    mock_boto3.client.return_value = MagicMock()
    mock_aioboto3.session.Session.return_value.client.return_value = MagicMock()

    caller = BedrockCaller(model=TEST_VISION_MODEL)

    llm_image = LLMImage(Url="https://example.com/image.png")
    message = LLMMessage(Role="user", Message="test", Images=[llm_image])

    with pytest.raises(NotImplementedError, match="does not support images by URL"):
        caller.format_message(message)


# ===== Messagelist Formatting Tests =====

@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_format_messagelist_basic(mock_aioboto3, mock_boto3):
    """Test formatting a basic conversation."""
    mock_boto3.client.return_value = MagicMock()
    mock_aioboto3.session.Session.return_value.client.return_value = MagicMock()

    caller = BedrockCaller(model=TEST_MODEL)
    messages = [
        LLMMessage(Role="user", Message="Hello"),
        LLMMessage(Role="assistant", Message="Hi"),
        LLMMessage(Role="user", Message="How are you?"),
    ]
    result = caller.format_messagelist(messages)

    assert len(result) == 3
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"
    assert result[2]["role"] == "user"


@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_format_messagelist_extracts_system(mock_aioboto3, mock_boto3):
    """Test that system messages are extracted to Defaults."""
    mock_boto3.client.return_value = MagicMock()
    mock_aioboto3.session.Session.return_value.client.return_value = MagicMock()

    caller = BedrockCaller(model=TEST_MODEL)
    messages = [
        LLMMessage(Role="system", Message="You are helpful"),
        LLMMessage(Role="user", Message="Hello"),
    ]
    result = caller.format_messagelist(messages)

    # System message should not be in result
    assert len(result) == 1
    assert result[0]["role"] == "user"

    # System message should be in Defaults
    assert caller.Defaults["system"] == [{"text": "You are helpful"}]


@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_format_messagelist_clears_system_when_absent(mock_aioboto3, mock_boto3):
    """Test that system is removed from Defaults when not in messages."""
    mock_boto3.client.return_value = MagicMock()
    mock_aioboto3.session.Session.return_value.client.return_value = MagicMock()

    caller = BedrockCaller(model=TEST_MODEL)

    # First set a system message
    caller.format_messagelist([
        LLMMessage(Role="system", Message="Old system"),
        LLMMessage(Role="user", Message="Test"),
    ])
    assert caller.Defaults.get("system") == [{"text": "Old system"}]

    # Then call without system message
    caller.format_messagelist([LLMMessage(Role="user", Message="Test")])
    assert "system" not in caller.Defaults


@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_format_messagelist_with_tool_messages(mock_aioboto3, mock_boto3):
    """Test formatting conversation with tool calls that get extended."""
    mock_boto3.client.return_value = MagicMock()
    mock_aioboto3.session.Session.return_value.client.return_value = MagicMock()

    caller = BedrockCaller(model=TEST_MODEL)

    tool_calls = [
        LLMToolCall(ID="call_1", Name="tool1", Response="result1"),
        LLMToolCall(ID="call_2", Name="tool2", Response="result2"),
    ]

    messages = [
        LLMMessage(Role="user", Message="test"),
        LLMMessage(Role="tool", ToolCalls=tool_calls),
    ]

    result = caller.format_messagelist(messages)

    # Should have 2 messages: user + tool results (extended)
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "user"  # Tool results sent as user in Bedrock


# ===== Tool Formatting Tests =====

@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_format_tool(mock_aioboto3, mock_boto3):
    """Test format_tool with Bedrock's toolSpec format."""
    mock_boto3.client.return_value = MagicMock()
    mock_aioboto3.session.Session.return_value.client.return_value = MagicMock()

    caller = BedrockCaller(model=TEST_MODEL)
    result = caller.format_tool(SampleTool())

    assert "toolSpec" in result
    assert result["toolSpec"]["name"] == "sample_tool"
    assert result["toolSpec"]["description"] == "A sample tool for testing"
    assert result["toolSpec"]["inputSchema"]["json"]["type"] == "object"
    assert "required_param" in result["toolSpec"]["inputSchema"]["json"]["properties"]
    assert "optional_param" in result["toolSpec"]["inputSchema"]["json"]["properties"]
    assert result["toolSpec"]["inputSchema"]["json"]["required"] == ["required_param"]


@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_format_tool_no_params(mock_aioboto3, mock_boto3):
    """Test format_tool with no parameters."""
    mock_boto3.client.return_value = MagicMock()
    mock_aioboto3.session.Session.return_value.client.return_value = MagicMock()

    class EmptyTool(Tool):
        name = "empty_tool"
        description = "No params"
        params = {}
        def execute(self, **p): return ""
        async def aexecute(self, **p): return ""

    caller = BedrockCaller(model=TEST_MODEL)
    result = caller.format_tool(EmptyTool())

    assert result["toolSpec"]["inputSchema"]["json"]["properties"] == {}
    assert result["toolSpec"]["inputSchema"]["json"]["required"] == []


# ===== Output Formatting Tests =====

@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_format_output_text_response(mock_aioboto3, mock_boto3):
    """Test format_output with text response."""
    mock_boto3.client.return_value = MagicMock()
    mock_aioboto3.session.Session.return_value.client.return_value = MagicMock()

    caller = BedrockCaller(model=TEST_MODEL)

    output = {
        "output": {
            "message": {
                "content": [{"text": "Test response"}]
            }
        },
        "stopReason": "end_turn"
    }

    result = caller.format_output(output, latency=1.5)

    assert isinstance(result, LLMMessage)
    assert result.Role == "assistant"
    assert result.Message == "Test response"
    assert result.Latency == 1.5


@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_format_output_text_with_leading_whitespace(mock_aioboto3, mock_boto3):
    """Test format_output strips leading whitespace from content."""
    mock_boto3.client.return_value = MagicMock()
    mock_aioboto3.session.Session.return_value.client.return_value = MagicMock()

    caller = BedrockCaller(model=TEST_MODEL)

    output = {
        "output": {
            "message": {
                "content": [{"text": "   Padded response"}]
            }
        },
        "stopReason": "end_turn"
    }

    result = caller.format_output(output)

    assert result.Message == "Padded response"


@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_format_output_with_tool_calls(mock_aioboto3, mock_boto3):
    """Test format_output with tool calls in response."""
    mock_boto3.client.return_value = MagicMock()
    mock_aioboto3.session.Session.return_value.client.return_value = MagicMock()

    caller = BedrockCaller(model=TEST_MODEL)

    output = {
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "call_abc",
                            "name": "get_weather",
                            "input": {"city": "NYC"}
                        }
                    },
                    {
                        "toolUse": {
                            "toolUseId": "call_def",
                            "name": "get_time",
                            "input": {"timezone": "UTC"}
                        }
                    }
                ]
            }
        },
        "stopReason": "tool_use"
    }

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
    assert result.Latency == 2.0


@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_format_output_generator(mock_aioboto3, mock_boto3):
    """Test format_output with generator type."""
    mock_boto3.client.return_value = MagicMock()
    mock_aioboto3.session.Session.return_value.client.return_value = MagicMock()

    caller = BedrockCaller(model=TEST_MODEL)

    def gen():
        yield "test"

    generator = gen()
    result = caller.format_output(generator)

    assert isinstance(result, GeneratorType)


# ===== _proc_call_args Tests =====

@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_proc_call_args_inference_config(mock_aioboto3, mock_boto3):
    """Test _proc_call_args creates inferenceConfig correctly."""
    mock_boto3.client.return_value = MagicMock()
    mock_aioboto3.session.Session.return_value.client.return_value = MagicMock()

    caller = BedrockCaller(model=TEST_MODEL)
    messages = [LLMMessage(Role="user", Message="test")]

    kwargs = caller._proc_call_args(messages, 100, None)

    assert "inferenceConfig" in kwargs
    assert kwargs["inferenceConfig"]["maxTokens"] == 100


@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_proc_call_args_with_temperature(mock_aioboto3, mock_boto3):
    """Test _proc_call_args moves temperature to inferenceConfig."""
    mock_boto3.client.return_value = MagicMock()
    mock_aioboto3.session.Session.return_value.client.return_value = MagicMock()

    caller = BedrockCaller(model=TEST_MODEL)
    messages = [LLMMessage(Role="user", Message="test")]

    kwargs = caller._proc_call_args(messages, 100, None, temperature=0.7)

    assert kwargs["inferenceConfig"]["temperature"] == 0.7
    assert "temperature" not in kwargs


@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_proc_call_args_with_top_p(mock_aioboto3, mock_boto3):
    """Test _proc_call_args moves topP to inferenceConfig."""
    mock_boto3.client.return_value = MagicMock()
    mock_aioboto3.session.Session.return_value.client.return_value = MagicMock()

    caller = BedrockCaller(model=TEST_MODEL)
    messages = [LLMMessage(Role="user", Message="test")]

    kwargs = caller._proc_call_args(messages, 100, None, topP=0.9)

    assert kwargs["inferenceConfig"]["topP"] == 0.9
    assert "topP" not in kwargs


@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_proc_call_args_with_stop_sequences(mock_aioboto3, mock_boto3):
    """Test _proc_call_args moves stopSequences to inferenceConfig."""
    mock_boto3.client.return_value = MagicMock()
    mock_aioboto3.session.Session.return_value.client.return_value = MagicMock()

    caller = BedrockCaller(model=TEST_MODEL)
    messages = [LLMMessage(Role="user", Message="test")]

    kwargs = caller._proc_call_args(messages, 100, None, stopSequences=["STOP"])

    assert kwargs["inferenceConfig"]["stopSequences"] == ["STOP"]
    assert "stopSequences" not in kwargs


@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_proc_call_args_with_tools(mock_aioboto3, mock_boto3):
    """Test _proc_call_args moves tools to toolConfig."""
    mock_boto3.client.return_value = MagicMock()
    mock_aioboto3.session.Session.return_value.client.return_value = MagicMock()

    caller = BedrockCaller(model=TEST_MODEL)
    messages = [LLMMessage(Role="user", Message="test")]

    tools = [{"toolSpec": {"name": "test_tool"}}]
    kwargs = caller._proc_call_args(messages, 100, None, tools=tools)

    assert "toolConfig" in kwargs
    assert kwargs["toolConfig"]["tools"] == tools
    assert "tools" not in kwargs


# ===== Tokenization Tests =====

@patch('fdllm.bedrock.caller.boto3')
@patch('fdllm.bedrock.caller.aioboto3')
def test_tokenize(mock_aioboto3, mock_boto3):
    """Test tokenize method."""
    mock_boto3.client.return_value = MagicMock()
    mock_aioboto3.session.Session.return_value.client.return_value = MagicMock()

    caller = BedrockCaller(model=TEST_MODEL)
    messages = [LLMMessage(Role="user", Message="Hello, world!")]

    tokens = caller.tokenize(messages)

    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_tokenize_bedrock_messages():
    """Test the tokenize_bedrock_messages helper function."""
    messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi there!"}]},
    ]

    tokens, mstr = tokenize_bedrock_messages(messages)

    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert isinstance(mstr, str)
    assert "Hello" in mstr
    assert "Hi there!" in mstr


def test_tokenize_bedrock_messages_empty_content():
    """Test tokenize_bedrock_messages with missing content."""
    messages = [
        {"role": "user", "content": []},
        {"role": "assistant"},  # No content key
    ]

    tokens, mstr = tokenize_bedrock_messages(messages)

    assert isinstance(tokens, list)


# ===== bedrock_async_wrapper Tests =====

@pytest.mark.anyio
async def test_bedrock_async_wrapper():
    """Test the bedrock_async_wrapper function."""
    # Create a mock async client
    mock_aclient = MagicMock()
    mock_context_manager = MagicMock()
    mock_client = MagicMock()
    mock_client.converse = MagicMock(return_value="test_result")

    # Set up async context manager
    async def async_enter():
        return mock_client
    mock_context_manager.__aenter__ = async_enter
    mock_context_manager.__aexit__ = MagicMock(return_value=None)
    mock_aclient.__aenter__ = async_enter
    mock_aclient.__aexit__ = MagicMock(return_value=None)

    # Make converse return a coroutine
    async def mock_converse(*args, **kwargs):
        return "test_result"
    mock_client.converse = mock_converse

    wrapper = bedrock_async_wrapper(mock_aclient)

    # The wrapper should be callable
    assert callable(wrapper)
