"""
Tests for Anthropic ClaudeCaller and ClaudeStreamingCaller.
Tests cover initialization, message formatting, tools, images, system messages, and output parsing.
"""
import pytest
import json
from types import SimpleNamespace, GeneratorType
from typing import List
from pathlib import Path
from unittest.mock import Mock, patch
from PIL import Image

from pydantic import BaseModel
from dotenv import load_dotenv

from fdllm import ClaudeCaller
from fdllm.anthropic import ClaudeStreamingCaller
from fdllm.llmtypes import LLMMessage, LLMToolCall, LLMImage
from fdllm.tooluse import Tool, ToolParam, ToolItem
from fdllm.sysutils import register_models

try:
    from anthropic.types.beta import BetaThinkingBlock, BetaToolUseBlock, BetaTextBlock
    from anthropic.types import Usage
except ImportError:
    pytest.skip("Anthropic SDK not installed", allow_module_level=True)

HERE = Path(__file__).resolve().parent
TEST_ROOT = HERE.parent

load_dotenv(TEST_ROOT / "test.env", override=True)


# ============================================================================
# Fixtures and Helpers
# ============================================================================

@pytest.fixture
def caller():
    """Create a ClaudeCaller instance for testing."""
    return ClaudeCaller()


@pytest.fixture
def vision_caller():
    """Create a ClaudeCaller with vision support."""
    return ClaudeCaller(model="claude-3-5-sonnet-latest")


def make_tool_block(id: str, name: str, input: dict = None):
    """Helper to create BetaToolUseBlock."""
    return BetaToolUseBlock(type="tool_use", id=id, name=name, input=input or {})


def make_text_block(text: str):
    """Helper to create BetaTextBlock."""
    return BetaTextBlock(text=text, type="text")


def make_output(content: list, input_tokens: int = 10, output_tokens: int = 10):
    """Helper to create mock API output."""
    return SimpleNamespace(
        content=content,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens)
    )


class SampleTool(Tool):
    """Reusable test tool with standard params."""
    name = "sample_tool"
    description = "A sample tool"
    params = {
        "required_param": ToolParam(type="string", description="Required", required=True),
        "optional_param": ToolParam(type="integer", description="Optional", default=0),
    }

    def execute(self, **params):
        return "result"

    async def aexecute(self, **params):
        return "result"


# ============================================================================
# Initialization Tests
# ============================================================================

class TestInitialization:
    """Tests for ClaudeCaller and ClaudeStreamingCaller initialization."""

    def test_init_default_model(self):
        """Test ClaudeCaller initializes with default model."""
        caller = ClaudeCaller()
        assert caller.Arg_Names.Messages == "messages"
        assert caller.Arg_Names.Model == "model"
        assert caller.Arg_Names.Response_Schema == "tools"
        assert caller.Model.Name == "claude-3-5-sonnet-latest"

    def test_init_specific_model(self):
        """Test ClaudeCaller with specific model."""
        caller = ClaudeCaller(model="claude-3-haiku-20240307")
        assert caller.Model.Name == "claude-3-haiku-20240307"

    def test_init_invalid_model(self):
        """Test ClaudeCaller rejects invalid model names."""
        with pytest.raises(NotImplementedError, match="not a recognised model name"):
            ClaudeCaller(model="invalid-model-name")

    def test_init_non_anthropic_model(self):
        """Test ClaudeCaller rejects non-Anthropic models."""
        with pytest.raises(ValueError, match="is not supported"):
            ClaudeCaller(model="gpt-4o")

    def test_init_streaming_caller(self):
        """Test ClaudeStreamingCaller initialization."""
        caller = ClaudeStreamingCaller()
        assert caller.Model.Name == "claude-3-5-sonnet-latest"
        # Streaming caller should have different retry methods
        assert caller._sync_call_with_retry is not None
        assert caller._async_call_with_retry is not None


# ============================================================================
# Message Formatting Tests
# ============================================================================

class TestFormatMessage:
    """Tests for format_message method."""

    def test_user_message(self, caller):
        """Test formatting user message."""
        msg = LLMMessage(Role="user", Message="Hello")
        assert caller.format_message(msg) == {"role": "user", "content": "Hello"}

    def test_assistant_message(self, caller):
        """Test formatting assistant message."""
        msg = LLMMessage(Role="assistant", Message="Hi there")
        assert caller.format_message(msg) == {"role": "assistant", "content": "Hi there"}

    def test_system_message(self, caller):
        """Test formatting system message."""
        msg = LLMMessage(Role="system", Message="You are helpful")
        assert caller.format_message(msg) == {"role": "system", "content": "You are helpful"}

    def test_tool_results(self, caller):
        """Test formatting tool results (tool role -> user role with tool_result)."""
        tool_calls = [
            LLMToolCall(ID="call_1", Name="tool1", Args={}, Response='{"result": 1}'),
            LLMToolCall(ID="call_2", Name="tool2", Args={}, Response='{"result": 2}'),
        ]
        msg = LLMMessage(Role="tool", ToolCalls=tool_calls)
        result = caller.format_message(msg)

        assert result["role"] == "user"
        assert len(result["content"]) == 2
        assert result["content"][0] == {
            "type": "tool_result",
            "tool_use_id": "call_1",
            "content": '{"result": 1}',
        }
        assert result["content"][1] == {
            "type": "tool_result",
            "tool_use_id": "call_2",
            "content": '{"result": 2}',
        }

    def test_assistant_with_tool_calls_and_text(self, caller):
        """Test formatting assistant message with both text and tool calls."""
        tool_calls = [
            LLMToolCall(ID="call_1", Name="search", Args={"q": "test"}),
        ]
        msg = LLMMessage(Role="assistant", Message="Searching...", ToolCalls=tool_calls)
        result = caller.format_message(msg)

        assert result["role"] == "assistant"
        assert len(result["content"]) == 2
        assert result["content"][0] == {"type": "text", "text": "Searching..."}
        assert result["content"][1] == {
            "type": "tool_use",
            "id": "call_1",
            "name": "search",
            "input": {"q": "test"},
        }

    def test_assistant_with_tool_calls_no_text(self, caller):
        """Test formatting assistant message with tool calls but no text."""
        tool_calls = [LLMToolCall(ID="call_1", Name="tool", Args={})]
        msg = LLMMessage(Role="assistant", ToolCalls=tool_calls)
        result = caller.format_message(msg)

        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "tool_use"

    def test_assistant_with_multiple_tool_calls(self, caller):
        """Test formatting assistant message with multiple tool calls."""
        tool_calls = [
            LLMToolCall(ID=f"call_{i}", Name=f"tool{i}", Args={"i": i})
            for i in range(3)
        ]
        msg = LLMMessage(Role="assistant", Message="Running", ToolCalls=tool_calls)
        result = caller.format_message(msg)

        assert len(result["content"]) == 4  # 1 text + 3 tools
        assert result["content"][0] == {"type": "text", "text": "Running"}
        for i, tc in enumerate(result["content"][1:]):
            assert tc["type"] == "tool_use"
            assert tc["id"] == f"call_{i}"


# ============================================================================
# Image Handling Tests
# ============================================================================

class TestImageHandling:
    """Tests for image handling in messages."""

    def test_single_image(self, vision_caller):
        """Test formatting message with single image."""
        img = Image.new('RGB', (10, 10), color='red')
        msg = LLMMessage(
            Role="user",
            Message="Analyze",
            Images=[LLMImage(Img=img)]
        )
        result = vision_caller.format_message(msg)

        assert result["role"] == "user"
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "image"
        assert result["content"][0]["source"]["type"] == "base64"
        assert result["content"][0]["source"]["media_type"] == "image/png"
        assert result["content"][1] == {"type": "text", "text": "Analyze"}

    def test_multiple_images(self, vision_caller):
        """Test formatting message with multiple images."""
        images = [
            LLMImage(Img=Image.new('RGB', (10, 10), color='red')),
            LLMImage(Img=Image.new('RGB', (10, 10), color='blue')),
        ]
        msg = LLMMessage(Role="user", Message="Compare", Images=images)
        result = vision_caller.format_message(msg)

        assert len(result["content"]) == 3  # 2 images + 1 text
        assert all(result["content"][i]["type"] == "image" for i in range(2))
        assert result["content"][2] == {"type": "text", "text": "Compare"}

    def test_image_url_not_supported(self, caller):
        """Test that image URLs raise NotImplementedError."""
        msg = LLMMessage(
            Role="user",
            Message="Test",
            Images=[LLMImage(Url="http://example.com/img.png")]
        )
        with pytest.raises(NotImplementedError, match="does not support images by URL"):
            caller.format_message(msg)

    def test_images_on_non_vision_model(self, vision_caller):
        """Test that images on non-vision model raises error."""
        vision_caller.Model.Vision = False
        msg = LLMMessage(
            Role="user",
            Message="Test",
            Images=[LLMImage(Img=Image.new('RGB', (10, 10)))]
        )
        with pytest.raises(NotImplementedError, match="doesn't support images"):
            vision_caller.format_message(msg)


# ============================================================================
# Messagelist Formatting Tests
# ============================================================================

class TestFormatMessagelist:
    """Tests for format_messagelist method."""

    def test_basic_conversation(self, caller):
        """Test formatting a basic conversation."""
        messages = [
            LLMMessage(Role="user", Message="Hello"),
            LLMMessage(Role="assistant", Message="Hi"),
            LLMMessage(Role="user", Message="How are you?"),
        ]
        result = caller.format_messagelist(messages)

        assert len(result) == 3
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi"}
        assert result[2] == {"role": "user", "content": "How are you?"}

    def test_extracts_system_message(self, caller):
        """Test that system messages are extracted to Defaults."""
        messages = [
            LLMMessage(Role="system", Message="You are helpful"),
            LLMMessage(Role="user", Message="Hello"),
        ]
        result = caller.format_messagelist(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert caller.Defaults["system"] == "You are helpful"

    def test_clears_system_when_absent(self, caller):
        """Test that system is removed from Defaults when not in messages."""
        # First set a system message
        caller.format_messagelist([
            LLMMessage(Role="system", Message="Old system"),
            LLMMessage(Role="user", Message="Test"),
        ])
        assert caller.Defaults.get("system") == "Old system"

        # Then call without system message
        caller.format_messagelist([LLMMessage(Role="user", Message="Test")])
        assert "system" not in caller.Defaults

    def test_with_tool_conversation(self, caller):
        """Test formatting conversation with tool calls and results."""
        messages = [
            LLMMessage(Role="user", Message="Search for X"),
            LLMMessage(Role="assistant", ToolCalls=[
                LLMToolCall(ID="c1", Name="search", Args={"q": "X"}),
            ]),
            LLMMessage(Role="tool", ToolCalls=[
                LLMToolCall(ID="c1", Name="search", Response="Found X"),
            ]),
        ]
        result = caller.format_messagelist(messages)

        assert len(result) == 3
        assert result[1]["role"] == "assistant"
        assert result[1]["content"][0]["type"] == "tool_use"
        assert result[2]["role"] == "user"  # tool results sent as user
        assert result[2]["content"][0]["type"] == "tool_result"


# ============================================================================
# Tool Formatting Tests
# ============================================================================

class TestToolFormatting:
    """Tests for format_tool and format_tools methods."""

    def test_format_tool_with_params(self, caller):
        """Test format_tool with required and optional params."""
        result = caller.format_tool(SampleTool())

        assert result["name"] == "sample_tool"
        assert result["description"] == "A sample tool"
        assert result["input_schema"]["type"] == "object"
        assert "required_param" in result["input_schema"]["properties"]
        assert "optional_param" in result["input_schema"]["properties"]
        assert result["input_schema"]["required"] == ["required_param"]

    def test_format_tool_no_params(self, caller):
        """Test format_tool with no parameters."""
        class EmptyTool(Tool):
            name = "empty"
            description = "No params"
            params = {}
            def execute(self, **p): return ""
            async def aexecute(self, **p): return ""

        result = caller.format_tool(EmptyTool())
        assert result["input_schema"]["properties"] == {}
        assert result["input_schema"]["required"] == []

    def test_format_tool_array_param(self, caller):
        """Test format_tool with array parameter."""
        class ArrayTool(Tool):
            name = "array_tool"
            description = "Has array"
            params = {
                "items": ToolParam(
                    type="array",
                    items=ToolItem(type="string"),
                    description="List",
                    required=True
                ),
            }
            def execute(self, **p): return ""
            async def aexecute(self, **p): return ""

        result = caller.format_tool(ArrayTool())
        assert result["input_schema"]["properties"]["items"]["type"] == "array"
        assert result["input_schema"]["properties"]["items"]["items"]["type"] == "string"

    def test_format_tool_enum_param(self, caller):
        """Test format_tool with enum parameter."""
        class EnumTool(Tool):
            name = "enum_tool"
            description = "Has enum"
            params = {
                "choice": ToolParam(
                    type="string",
                    enum=["a", "b", "c"],
                    description="Pick one",
                    required=True
                ),
            }
            def execute(self, **p): return ""
            async def aexecute(self, **p): return ""

        result = caller.format_tool(EnumTool())
        assert result["input_schema"]["properties"]["choice"]["enum"] == ["a", "b", "c"]

    def test_format_tools_multiple(self, caller):
        """Test format_tools with multiple tools."""
        class Tool1(Tool):
            name = "tool1"
            description = "First"
            params = {}
            def execute(self, **p): return ""
            async def aexecute(self, **p): return ""

        class Tool2(Tool):
            name = "tool2"
            description = "Second"
            params = {}
            def execute(self, **p): return ""
            async def aexecute(self, **p): return ""

        result = caller.format_tools([Tool1(), Tool2()])
        assert len(result) == 2
        assert result[0]["name"] == "tool1"
        assert result[1]["name"] == "tool2"


# ============================================================================
# Output Formatting Tests
# ============================================================================

class TestFormatOutput:
    """Tests for format_output method."""

    def test_text_response(self, caller):
        """Test format_output with simple text response."""
        output = make_output([make_text_block("Hello")])
        result = caller.format_output(output, latency=1.5)

        assert isinstance(result, LLMMessage)
        assert result.Role == "assistant"
        assert result.Message == "Hello"
        assert result.TokensUsed == 20
        assert result.TokensUsedCompletion == 10
        assert result.Latency == 1.5

    def test_single_tool_call(self, caller):
        """Test format_output with single tool call."""
        output = make_output([make_tool_block("c1", "search", {"q": "test"})])
        result = caller.format_output(output)

        assert result.Message == ""
        assert len(result.ToolCalls) == 1
        assert result.ToolCalls[0].ID == "c1"
        assert result.ToolCalls[0].Name == "search"
        assert result.ToolCalls[0].Args == {"q": "test"}

    def test_multiple_tool_calls(self, caller):
        """Test format_output with multiple tool calls."""
        output = make_output([
            make_tool_block(f"c{i}", f"tool{i}", {"i": i})
            for i in range(3)
        ])
        result = caller.format_output(output)

        assert len(result.ToolCalls) == 3
        for i in range(3):
            assert result.ToolCalls[i].ID == f"c{i}"
            assert result.ToolCalls[i].Name == f"tool{i}"
            assert result.ToolCalls[i].Args == {"i": i}

    def test_text_followed_by_tool_calls(self, caller):
        """Test format_output with text then tool calls."""
        output = make_output([
            make_text_block("Let me search..."),
            make_tool_block("c1", "search", {}),
            make_tool_block("c2", "lookup", {}),
        ])
        result = caller.format_output(output)

        assert result.Message == "Let me search..."
        assert len(result.ToolCalls) == 2

    def test_with_thinking_block(self, caller):
        """Test format_output with BetaThinkingBlock (reasoning)."""
        thinking = BetaThinkingBlock(type="thinking", thinking="Hmm...", signature="sig")
        output = SimpleNamespace(
            content=[thinking, make_text_block("Answer")],
            usage=Usage(input_tokens=10, output_tokens=30)
        )

        with patch.object(caller.Client.beta.messages, 'count_tokens',
                          return_value=SimpleNamespace(input_tokens=5)):
            result = caller.format_output(output)

        assert result.Message == "Answer"
        assert result.TokensUsedReasoning == 5

    def test_response_schema_structured_output(self, caller):
        """Test format_output with response_schema returns JSON string."""
        class Schema(BaseModel):
            name: str
            value: int

        output = make_output([
            make_tool_block("c1", "Schema", {"name": "test", "value": 42})
        ])
        result = caller.format_output(output, response_schema=Schema)

        assert result.ToolCalls is None
        assert json.loads(result.Message) == {"name": "test", "value": 42}

    def test_no_usage_metadata(self, caller):
        """Test format_output when usage is None."""
        output = SimpleNamespace(
            content=[make_text_block("Hello")],
            usage=None
        )
        result = caller.format_output(output)

        assert result.Message == "Hello"
        assert result.TokensUsed is None
        assert result.TokensUsedCompletion is None

    def test_generator_passthrough(self, caller):
        """Test format_output passes through generators."""
        def gen():
            yield "test"
        result = caller.format_output(gen())
        assert isinstance(result, GeneratorType)

    def test_empty_content_raises(self, caller):
        """Test format_output with empty content list raises IndexError."""
        output = make_output([])
        with pytest.raises(IndexError):
            caller.format_output(output)

    def test_none_content_raises(self, caller):
        """Test format_output with None content raises UnboundLocalError."""
        output = SimpleNamespace(content=None)
        with pytest.raises(UnboundLocalError):
            caller.format_output(output)


# ============================================================================
# Token Counting Tests
# ============================================================================

class TestTokenCounting:
    """Tests for count_tokens method."""

    def test_count_tokens(self, caller):
        """Test count_tokens calls API correctly."""
        caller.Client.beta.messages.count_tokens = Mock(
            return_value=SimpleNamespace(input_tokens=150)
        )
        messages = [LLMMessage(Role="user", Message="Hello")]

        count = caller.count_tokens(messages)

        assert count == 150
        caller.Client.beta.messages.count_tokens.assert_called_once()


# ============================================================================
# _proc_call_args Tests
# ============================================================================

class TestProcCallArgs:
    """Tests for _proc_call_args method."""

    def test_with_response_schema(self, caller):
        """Test _proc_call_args converts response_schema to tool."""
        class Schema(BaseModel):
            field: str

        messages = [LLMMessage(Role="user", Message="test")]
        kwargs = caller._proc_call_args(messages, 100, Schema)

        assert "tools" in kwargs
        assert kwargs["tools"][0]["name"] == "Schema"
        assert kwargs["tool_choice"] == {"type": "tool", "name": "Schema"}

    def test_with_nested_response_schema(self, caller):
        """Test _proc_call_args resolves $ref in nested schemas."""
        class Inner(BaseModel):
            value: int

        class Outer(BaseModel):
            items: List[Inner]

        messages = [LLMMessage(Role="user", Message="test")]
        kwargs = caller._proc_call_args(messages, 100, Outer)

        assert kwargs["tools"][0]["name"] == "Outer"
        # Should not crash on $ref resolution

    def test_without_response_schema(self, caller):
        """Test _proc_call_args without response_schema."""
        messages = [LLMMessage(Role="user", Message="test")]
        kwargs = caller._proc_call_args(messages, 100, None)

        assert "tool_choice" not in kwargs
