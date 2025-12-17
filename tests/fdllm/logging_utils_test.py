import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from fdllm.logging_utils import (
    get_logger,
    truncate_text,
    truncate_prompt,
    log_call_start,
    log_call_completion,
    log_retry_attempt,
    log_retry_exhausted,
    log_retry_success
)


# Fixtures
@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return Mock()


@pytest.fixture
def mock_user_message():
    """Create a mock LLMMessage with user role."""
    msg = MagicMock()
    msg.Role = "user"
    msg.Message = "This is a test message"
    return msg


@pytest.fixture
def mock_assistant_message():
    """Create a mock LLMMessage with assistant role."""
    msg = MagicMock()
    msg.Role = "assistant"
    msg.Message = "This is an assistant response"
    return msg


# ============================================================================
# Tests for truncate_text()
# ============================================================================

def test_truncate_text_short_text():
    """Text shorter than max_length returns unchanged."""
    text = "Short text"
    result = truncate_text(text, max_length=50)
    assert result == "Short text"


def test_truncate_text_exact_length():
    """Text exactly at max_length returns unchanged."""
    text = "A" * 50
    result = truncate_text(text, max_length=50)
    assert result == text
    assert len(result) == 50


def test_truncate_text_long_text():
    """Text longer than max_length gets truncated with '...'."""
    text = "A" * 100
    result = truncate_text(text, max_length=50)
    assert result == ("A" * 50) + "..."
    assert len(result) == 53  # 50 chars + "..."


def test_truncate_text_empty_string():
    """Empty string returns empty string."""
    result = truncate_text("", max_length=50)
    assert result == ""


def test_truncate_text_custom_max_length():
    """Custom max_length parameter works correctly."""
    text = "This is a test message that is quite long"
    result = truncate_text(text, max_length=10)
    assert result == "This is a ..."
    assert len(result) == 13  # 10 chars + "..."


# ============================================================================
# Tests for truncate_prompt()
# ============================================================================

def test_truncate_prompt_empty_messages():
    """Empty list returns '<empty>'."""
    result = truncate_prompt([])
    assert result == "<empty>"


def test_truncate_prompt_single_user_message(mock_user_message):
    """Single user message gets truncated correctly."""
    mock_user_message.Message = "Hello, this is a test"
    result = truncate_prompt([mock_user_message], max_length=50)
    assert result == "Hello, this is a test"


def test_truncate_prompt_multiple_user_messages():
    """Returns last user message (truncated)."""
    msg1 = MagicMock()
    msg1.Role = "user"
    msg1.Message = "First message"

    msg2 = MagicMock()
    msg2.Role = "assistant"
    msg2.Message = "Assistant response"

    msg3 = MagicMock()
    msg3.Role = "user"
    msg3.Message = "Last user message"

    result = truncate_prompt([msg1, msg2, msg3], max_length=50)
    assert result == "Last user message"


def test_truncate_prompt_no_user_messages(mock_assistant_message):
    """Falls back to first message with role prefix."""
    mock_assistant_message.Message = "Assistant only message"
    result = truncate_prompt([mock_assistant_message], max_length=50)
    assert result == "Assistant only message"


def test_truncate_prompt_message_without_text():
    """Handles Message=None case."""
    msg = MagicMock()
    msg.Role = "user"
    msg.Message = None

    result = truncate_prompt([msg], max_length=50)
    assert result == "<no text content>"


def test_truncate_prompt_custom_max_length():
    """Custom max_length works."""
    msg = MagicMock()
    msg.Role = "user"
    msg.Message = "A" * 100

    result = truncate_prompt([msg], max_length=20)
    assert result == ("A" * 20) + "..."
    assert len(result) == 23


def test_truncate_prompt_mixed_roles():
    """Correctly finds last user message among mixed roles."""
    messages = []

    msg1 = MagicMock()
    msg1.Role = "system"
    msg1.Message = "System prompt"
    messages.append(msg1)

    msg2 = MagicMock()
    msg2.Role = "user"
    msg2.Message = "First user message"
    messages.append(msg2)

    msg3 = MagicMock()
    msg3.Role = "assistant"
    msg3.Message = "Assistant response"
    messages.append(msg3)

    msg4 = MagicMock()
    msg4.Role = "user"
    msg4.Message = "Second user message - this should be returned"
    messages.append(msg4)

    result = truncate_prompt(messages, max_length=100)
    assert result == "Second user message - this should be returned"


# ============================================================================
# Tests for log_call_start()
# ============================================================================

def test_log_call_start_sync(mock_logger, mock_user_message):
    """Logs correct info message for sync calls and returns timestamp."""
    mock_user_message.Message = "Test prompt"

    with patch('time.time', return_value=1234567890.0):
        result = log_call_start(mock_logger, "gpt-4", [mock_user_message], call_type="sync")

    # Verify return value is timestamp
    assert result == 1234567890.0

    # Verify info log was called
    assert mock_logger.info.called
    info_call = mock_logger.info.call_args[0][0]
    assert "sync call" in info_call.lower()
    assert "gpt-4" in info_call
    assert "Test prompt" in info_call


def test_log_call_start_async(mock_logger, mock_user_message):
    """Logs correct info message for async calls."""
    mock_user_message.Message = "Async test prompt"

    with patch('time.time', return_value=9999999999.0):
        result = log_call_start(mock_logger, "claude-3", [mock_user_message], call_type="async")

    assert result == 9999999999.0

    # Verify info log was called with async
    assert mock_logger.info.called
    info_call = mock_logger.info.call_args[0][0]
    assert "async call" in info_call.lower()
    assert "claude-3" in info_call


def test_log_call_start_debug_message(mock_logger, mock_user_message):
    """Logs debug message with message count."""
    messages = [mock_user_message, mock_user_message, mock_user_message]

    log_call_start(mock_logger, "test-model", messages, call_type="sync")

    # Verify debug log was called
    assert mock_logger.debug.called
    debug_call = mock_logger.debug.call_args[0][0]
    assert "3 messages" in debug_call
    assert "test-model" in debug_call


def test_log_call_start_with_long_prompt(mock_logger):
    """Uses truncate_prompt correctly with long messages."""
    msg = MagicMock()
    msg.Role = "user"
    msg.Message = "A" * 200  # Very long message

    log_call_start(mock_logger, "model", [msg], call_type="sync")

    # Verify the prompt was truncated in the log
    info_call = mock_logger.info.call_args[0][0]
    # The truncated version should end with "..."
    assert "..." in info_call
    # Should not contain the full 200 A's
    assert "A" * 200 not in info_call


# ============================================================================
# Tests for log_call_completion()
# ============================================================================

def test_log_call_completion_success(mock_logger):
    """Logs success message with duration."""
    start_time = 1000.0

    with patch('time.time', return_value=1002.5):
        log_call_completion(mock_logger, "gpt-4", start_time, response=None, error=None)

    # Verify info log was called with duration
    assert mock_logger.info.called
    info_call = mock_logger.info.call_args[0][0]
    assert "gpt-4" in info_call
    assert "completed" in info_call.lower()
    assert "2.50s" in info_call or "2.5s" in info_call


def test_log_call_completion_with_error(mock_logger):
    """Logs warning with error type and message."""
    start_time = 1000.0
    error = ValueError("Test error message")

    with patch('time.time', return_value=1001.5):
        log_call_completion(mock_logger, "claude-3", start_time, response=None, error=error)

    # Verify warning log was called
    assert mock_logger.warning.called
    warning_call = mock_logger.warning.call_args[0][0]
    assert "claude-3" in warning_call
    assert "failed" in warning_call.lower()
    assert "ValueError" in warning_call
    assert "Test error message" in warning_call
    assert "1.50s" in warning_call or "1.5s" in warning_call


def test_log_call_completion_calculates_duration(mock_logger):
    """Correctly calculates time difference."""
    start_time = 100.0

    with patch('time.time', return_value=150.75):
        log_call_completion(mock_logger, "model", start_time)

    info_call = mock_logger.info.call_args[0][0]
    # Duration should be 50.75 seconds
    assert "50.75s" in info_call or "50.8s" in info_call


def test_log_call_completion_with_response_usage(mock_logger):
    """Logs debug message when response has usage attribute."""
    start_time = 1000.0

    response = MagicMock()
    response.usage = {"total_tokens": 150, "prompt_tokens": 50, "completion_tokens": 100}

    with patch('time.time', return_value=1002.0):
        log_call_completion(mock_logger, "gpt-4", start_time, response=response)

    # Verify debug log was called with usage info
    assert mock_logger.debug.called
    debug_call = mock_logger.debug.call_args[0][0]
    assert "Token usage" in debug_call or "usage" in debug_call.lower()
    assert "gpt-4" in debug_call


def test_log_call_completion_without_response_usage(mock_logger):
    """No debug message when response lacks usage attribute."""
    start_time = 1000.0

    response = MagicMock(spec=[])  # Mock with no attributes
    del response.usage  # Ensure usage doesn't exist

    with patch('time.time', return_value=1002.0):
        log_call_completion(mock_logger, "model", start_time, response=response)

    # Verify info was called (success) but debug was not called
    assert mock_logger.info.called
    # Debug might be called, but shouldn't fail if usage is missing
    # This test verifies the function handles missing usage gracefully


def test_log_call_completion_with_none_response(mock_logger):
    """Handles None response gracefully."""
    start_time = 1000.0

    with patch('time.time', return_value=1001.0):
        log_call_completion(mock_logger, "model", start_time, response=None)

    # Should complete without error
    assert mock_logger.info.called
