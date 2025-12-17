"""
Tests for Google GenAI caller.
Tests cover message formatting, tools, images, system message handling, and output parsing.
"""
import pytest
from types import SimpleNamespace, GeneratorType
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from dotenv import load_dotenv

from fdllm import GoogleGenAICaller
from fdllm.llmtypes import LLMMessage, LLMToolCall, LLMImage
from fdllm.tooluse import Tool, ToolParam
from fdllm.sysutils import register_models

HERE = Path(__file__).resolve().parent
TEST_ROOT = HERE.parent

load_dotenv(TEST_ROOT / "test.env", override=True)

register_models(TEST_ROOT / "custom_models_test.yaml")


# ===== Basic GoogleGenAICaller Tests =====

def test_init_google():
    """Test GoogleGenAICaller initialization"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")
    assert caller.Arg_Names.Messages == "contents"
    assert caller.Arg_Names.Model == "model"
    assert caller.Arg_Names.Response_Schema == "response_schema"


def test_format_message_basic_roles():
    """Test formatting basic user and system messages"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")

    # User message
    user_msg = LLMMessage(Role="user", Message="Hello")
    result = caller.format_message(user_msg)
    assert result == {"role": "user", "parts": [{"text": "Hello"}]}

    # System message (treated as user in format_message)
    system_msg = LLMMessage(Role="system", Message="You are helpful")
    result = caller.format_message(system_msg)
    assert result == {"role": "system", "parts": [{"text": "You are helpful"}]}


def test_format_message_assistant_to_model():
    """Test that assistant role is converted to model role"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")

    message = LLMMessage(Role="assistant", Message="I can help")
    result = caller.format_message(message)

    assert result["role"] == "model"
    assert result["parts"] == [{"text": "I can help"}]


# ===== Tool Handling Tests =====

def test_format_message_tool_role():
    """Test formatting of messages with tool role"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")

    tool_calls = [
            LLMToolCall(ID="call_123", Name="get_weather", Args={"city": "NYC"}, Response='{"temp": 72}'),
            LLMToolCall(ID="call_456", Name="get_time", Args={}, Response='{"time": "12:00"}'),
        ]

    message = LLMMessage(Role="tool", ToolCalls=tool_calls)
    result = caller.format_message(message)

    assert result["role"] == "tool"
    assert len(result["parts"]) == 2
    assert result["parts"][0]["function_response"]["id"] == "call_123"
    assert result["parts"][0]["function_response"]["name"] == "get_weather"
    assert result["parts"][0]["function_response"]["response"] == {"result": '{"temp": 72}'}


def test_format_message_assistant_tool_calls():
    """Test formatting of assistant messages with tool calls"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")

    tool_calls = [
            LLMToolCall(ID="call_789", Name="search", Args={"query": "test"}),
        ]

    message = LLMMessage(Role="assistant", ToolCalls=tool_calls)
    result = caller.format_message(message)

    assert result["role"] == "model"
    assert len(result["parts"]) == 1
    assert result["parts"][0]["function_call"]["id"] == "call_789"
    assert result["parts"][0]["function_call"]["name"] == "search"
    assert result["parts"][0]["function_call"]["args"] == {"query": "test"}


def test_format_tool():
    """Test format_tool method with Google's format"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")

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

    assert "function_declarations" in result
    assert len(result["function_declarations"]) == 1
    func_decl = result["function_declarations"][0]
    assert func_decl["name"] == "test_tool"
    assert func_decl["description"] == "A test tool"
    assert func_decl["parameters"]["type"] == "OBJECT"
    assert func_decl["parameters"]["properties"]["param1"]["type"] == "STRING"
    assert func_decl["parameters"]["properties"]["param2"]["type"] == "INTEGER"
    assert func_decl["parameters"]["required"] == ["param1"]


def test_format_tool_no_params():
    """Test format_tool with no parameters"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")

    class SimpleTool(Tool):
            name = "simple_tool"
            description = "Simple"
            params = {}

            def execute(self, **params):
                return "result"

            async def aexecute(self, **params):
                return "result"

    tool = SimpleTool()
    result = caller.format_tool(tool)

    assert "function_declarations" in result
    func_decl = result["function_declarations"][0]
    assert func_decl["name"] == "simple_tool"
    assert "parameters" not in func_decl


# ===== Image Handling Tests =====

def test_format_message_with_images():
    """Test formatting of user messages with images"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")
    caller.Model.Vision = True

    img = Image.new('RGB', (10, 10), color='red')
    llm_image = LLMImage(Img=img)

    message = LLMMessage(Role="user", Message="What's in this image?", Images=[llm_image])
    result = caller.format_message(message)

    assert result["role"] == "user"
    assert len(result["parts"]) == 2  # image + text
    assert "inline_data" in result["parts"][0]
    assert result["parts"][0]["inline_data"]["mime_type"] == "image/png"
    assert "data" in result["parts"][0]["inline_data"]
    assert result["parts"][1] == {"text": "What's in this image?"}


def test_format_message_images_non_vision_model():
    """Test that images raise error for non-vision models"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")
    caller.Model.Vision = False

    img = Image.new('RGB', (10, 10))
    llm_image = LLMImage(Img=img)
    message = LLMMessage(Role="user", Message="test", Images=[llm_image])

    with pytest.raises(NotImplementedError, match="doesn't support images"):
            caller.format_message(message)


def test_format_message_multiple_images():
    """Test formatting with multiple images"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")
    caller.Model.Vision = True

    img1 = Image.new('RGB', (10, 10), color='red')
    img2 = Image.new('RGB', (10, 10), color='blue')
    llm_images = [LLMImage(Img=img1), LLMImage(Img=img2)]

    message = LLMMessage(Role="user", Message="Compare these", Images=llm_images)
    result = caller.format_message(message)

    assert len(result["parts"]) == 3  # 2 images + text
    assert all("inline_data" in part for part in result["parts"][:2])


# ===== System Message Handling Tests (Google-Specific) =====

def test_format_messagelist_extracts_system_message():
    """Test that system messages are extracted to Defaults"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")

    messages = [
            LLMMessage(Role="system", Message="You are helpful"),
            LLMMessage(Role="user", Message="Hello"),
            LLMMessage(Role="assistant", Message="Hi there"),
        ]

    result = caller.format_messagelist(messages)

        # System message should NOT be in result
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "model"

        # System message should be in Defaults
    assert caller.Defaults["system"] == "You are helpful"


def test_format_messagelist_multiple_system_messages():
    """Test that only first system message is used"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")

    messages = [
            LLMMessage(Role="system", Message="First system"),
            LLMMessage(Role="user", Message="Hello"),
            LLMMessage(Role="system", Message="Second system"),
        ]

    result = caller.format_messagelist(messages)

        # Only first system message should be stored
    assert caller.Defaults["system"] == "First system"


def test_format_messagelist_no_system_message():
    """Test that system key is removed when no system message"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")
    caller.Defaults["system"] = "Old system message"

    messages = [
            LLMMessage(Role="user", Message="Hello"),
        ]

    result = caller.format_messagelist(messages)

        # System should be removed from Defaults
    assert "system" not in caller.Defaults


# ===== Config Processing Tests (_proc_call_args) =====

def test_proc_call_args_system_instruction():
    """Test that system instruction is moved to config"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")

    messages = [LLMMessage(Role="user", Message="test")]
    kwargs = caller._proc_call_args(messages, 100, None, system="You are helpful")

    assert "config" in kwargs
    assert kwargs["config"]["system_instruction"] == "You are helpful"
    assert "system" not in kwargs


def test_proc_call_args_logprobs():
    """Test that logprobs are renamed for Google"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")

    messages = [LLMMessage(Role="user", Message="test")]
    kwargs = caller._proc_call_args(messages, 100, None, logprobs=True, top_logprobs=5)

    assert kwargs["config"]["response_logprobs"] == True
    assert kwargs["config"]["logprobs"] == 5
    assert "logprobs" not in kwargs or kwargs.get("logprobs") is None
    assert "top_logprobs" not in kwargs or kwargs.get("top_logprobs") is None


def test_proc_call_args_response_schema():
    """Test that response_schema sets mime type"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")
    from pydantic import BaseModel

    class Schema(BaseModel):
        result: str

    messages = [LLMMessage(Role="user", Message="test")]
    kwargs = caller._proc_call_args(messages, 100, Schema)

    assert kwargs["config"]["response_schema"] is not None
    assert kwargs["config"]["response_mime_type"] == "application/json"


def test_proc_call_args_no_response_schema():
    """Test that response_schema is removed when None"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")

    messages = [LLMMessage(Role="user", Message="test")]
    kwargs = caller._proc_call_args(messages, 100, None)

    assert "response_schema" not in kwargs["config"]


def test_proc_call_args_moves_params_to_config():
    """Test that various parameters are moved to config dict"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")

    messages = [LLMMessage(Role="user", Message="test")]
    kwargs = caller._proc_call_args(
            messages, 100, None,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            seed=42,
        )

    config = kwargs["config"]
    assert config["temperature"] == 0.7
    assert config["top_p"] == 0.9
    assert config["top_k"] == 40
    assert config["seed"] == 42
    assert "temperature" not in kwargs or kwargs.get("temperature") is None


# ===== Output Formatting Tests =====

def test_format_output_text_response():
    """Test format_output with text response"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")

        # Mock Google's response structure
    part = SimpleNamespace(text="Test response", function_call=None)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content, logprobs_result=None)
    output = SimpleNamespace(
            candidates=[candidate],
            usage_metadata=SimpleNamespace(
                total_token_count=100,
                candidates_token_count=50,
                thoughts_token_count=None,
            )
        )

    result = caller.format_output(output, latency=1.5)

    assert isinstance(result, LLMMessage)
    assert result.Role == "assistant"
    assert result.Message == "Test response"
    assert result.TokensUsed == 100
    assert result.TokensUsedCompletion == 50
    assert result.Latency == 1.5


def test_format_output_with_reasoning_tokens():
    """Test format_output with thoughts_token_count (reasoning)"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")

    part = SimpleNamespace(text="Answer", function_call=None)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content, logprobs_result=None)
    output = SimpleNamespace(
            candidates=[candidate],
            usage_metadata=SimpleNamespace(
                total_token_count=150,
                candidates_token_count=50,
                thoughts_token_count=100,  # Reasoning tokens
            )
        )

    result = caller.format_output(output)

    assert result.TokensUsedReasoning == 100


def test_format_output_without_usage_metadata():
    """Test format_output when usage_metadata is None"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")

    part = SimpleNamespace(text="Response", function_call=None)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content, logprobs_result=None)
    output = SimpleNamespace(
            candidates=[candidate],
            usage_metadata=None,
        )

    result = caller.format_output(output)

    assert result.Message == "Response"
        # Should not crash, token counts won't be set


def test_format_output_with_tool_calls():
    """Test format_output with function calls"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")

    part = SimpleNamespace(
            text=None,
            function_call=SimpleNamespace(
                id="call_abc",
                name="get_data",
                args={"param": "value"}
            )
        )
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content, logprobs_result=None)
    output = SimpleNamespace(
            candidates=[candidate],
            usage_metadata=SimpleNamespace(
                total_token_count=75,
                candidates_token_count=25,
                thoughts_token_count=None,
            )
        )

    result = caller.format_output(output)

    assert isinstance(result, LLMMessage)
    assert result.Role == "assistant"
    assert result.ToolCalls is not None
    assert len(result.ToolCalls) == 1
    assert result.ToolCalls[0].ID == "call_abc"
    assert result.ToolCalls[0].Name == "get_data"
    assert result.ToolCalls[0].Args == {"param": "value"}


def test_format_output_multiple_tool_calls():
    """Test format_output with multiple function calls"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")

    parts = [
            SimpleNamespace(
                function_call=SimpleNamespace(id="call_1", name="tool1", args={})
            ),
            SimpleNamespace(
                function_call=SimpleNamespace(id="call_2", name="tool2", args={})
            ),
        ]
    content = SimpleNamespace(parts=parts)
    candidate = SimpleNamespace(content=content, logprobs_result=None)
    output = SimpleNamespace(candidates=[candidate], usage_metadata=None)

    result = caller.format_output(output)

    assert len(result.ToolCalls) == 2


def test_format_output_with_logprobs():
    """Test format_output with logprobs conversion"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")

        # Mock Google's logprobs structure
    chosen_candidate = SimpleNamespace(token="hello", log_probability=-0.5)
    top_candidate = SimpleNamespace(
            candidates=[
                SimpleNamespace(token="hello", log_probability=-0.5),
                SimpleNamespace(token="hi", log_probability=-1.0),
            ]
        )

    part = SimpleNamespace(text="hello", function_call=None)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(
            content=content,
            logprobs_result=SimpleNamespace(
                chosen_candidates=[chosen_candidate],
                top_candidates=[top_candidate],
            )
        )
    output = SimpleNamespace(candidates=[candidate], usage_metadata=None)

    result = caller.format_output(output)

    assert result.LogProbs is not None
    assert len(result.LogProbs.content) == 1
    assert result.LogProbs.content[0].token == "hello"
    assert result.LogProbs.content[0].logprob == -0.5
    assert len(result.LogProbs.content[0].top_logprobs) == 2


def test_format_output_invalid():
    """Test format_output with invalid output"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")

        # Part with neither text nor function_call
    part = SimpleNamespace(text=None, function_call=None)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content, logprobs_result=None)
    output = SimpleNamespace(candidates=[candidate], usage_metadata=None)

    with pytest.raises(ValueError, match="Output must be either content or tool call"):
            caller.format_output(output)


def test_format_output_generator():
    """Test format_output with generator type"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")

    def gen():
            yield "test"

    generator = gen()
    result = caller.format_output(generator)

    assert isinstance(result, GeneratorType)


# ===== Token Counting Tests =====

def test_count_tokens():
    """Test count_tokens API call"""
    caller = GoogleGenAICaller(model="gemini-2.0-flash")

    # Mock the client's count_tokens method
    mock_response = SimpleNamespace(total_tokens=150)
    caller.Client.models.count_tokens = Mock(return_value=mock_response)

    messages = [LLMMessage(Role="user", Message="Hello")]

    token_count = caller.count_tokens(messages)

    assert token_count == 150
    caller.Client.models.count_tokens.assert_called_once()
    call_args = caller.Client.models.count_tokens.call_args
    assert call_args.kwargs["model"] == caller.Model.Api_Model_Name
    assert "contents" in call_args.kwargs
