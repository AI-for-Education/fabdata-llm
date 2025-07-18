"""
Logging utilities for the FabData-LLM library.

Provides centralized logging configuration and helper functions for safe,
privacy-conscious logging of LLM interactions.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .llmtypes import LLMMessage


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module name.
    
    Args:
        name: Module name, typically __name__
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"fdllm.{name}")


def truncate_text(text: str, max_length: int = 50) -> str:
    """
    Safely truncate text for logging, avoiding sensitive data exposure.
    
    Args:
        text: Text to truncate
        max_length: Maximum length to keep
        
    Returns:
        Truncated text with ellipsis if needed
    """
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."


def truncate_prompt(messages: List["LLMMessage"], max_length: int = 50) -> str:
    """
    Extract and truncate the main prompt content for logging.
    
    Args:
        messages: List of LLM messages
        max_length: Maximum length to keep
        
    Returns:
        Truncated representation of the prompt
    """
    if not messages:
        return "<empty>"
    
    # Find the last user message for the most relevant content
    user_messages = [msg for msg in messages if msg.Role == "user"]
    if user_messages:
        last_user_msg = user_messages[-1]
        content = last_user_msg.Message or "<no text content>"
    else:
        # Fallback to first message
        content = messages[0].Message or f"<{messages[0].Role} message>"
    
    return truncate_text(content, max_length)


def log_call_start(logger: logging.Logger, model: str, messages: List["LLMMessage"], 
                   call_type: str = "sync") -> float:
    """
    Log the start of an LLM API call.
    
    Args:
        logger: Logger instance
        model: Model name being called
        messages: Message list
        call_type: "sync" or "async"
        
    Returns:
        Start timestamp for duration calculation
    """
    start_time = time.time()
    prompt_preview = truncate_prompt(messages)
    
    logger.info(
        f"Starting {call_type} call to {model} - prompt: {prompt_preview}"
    )
    
    logger.debug(
        f"Call details - {call_type} call to {model}, {len(messages)} messages"
    )
    
    return start_time


def log_call_completion(logger: logging.Logger, model: str, start_time: float, 
                       response: Optional[Any] = None, error: Optional[Exception] = None):
    """
    Log the completion of an LLM API call.
    
    Args:
        logger: Logger instance
        model: Model name that was called
        start_time: Start timestamp from log_call_start
        response: Response object (optional)
        error: Exception if call failed (optional)
    """
    duration = time.time() - start_time
    
    if error:
        logger.warning(
            f"Call to {model} failed after {duration:.2f}s - {type(error).__name__}: {str(error)}"
        )
    else:
        logger.info(
            f"Call to {model} completed in {duration:.2f}s"
        )
        
        # Log additional metadata if available
        if response and hasattr(response, 'usage'):
            logger.debug(
                f"Token usage for {model}: {response.usage}"
            )


def log_retry_attempt(logger: logging.Logger, attempt: int, max_attempts: int, 
                      error: Exception, wait_time: float, total_elapsed: float):
    """
    Log a retry attempt with detailed information.
    
    Args:
        logger: Logger instance
        attempt: Current attempt number (1-indexed)
        max_attempts: Maximum number of attempts
        error: Exception that triggered the retry
        wait_time: Time to wait before next attempt
        total_elapsed: Total time elapsed so far
    """
    logger.warning(
        f"Retry {attempt}/{max_attempts} - {type(error).__name__}: {str(error)} "
        f"(waiting {wait_time:.1f}s, elapsed: {total_elapsed:.1f}s)"
    )


def log_retry_exhausted(logger: logging.Logger, max_attempts: int, 
                       final_error: Exception, total_elapsed: float):
    """
    Log when all retry attempts have been exhausted.
    
    Args:
        logger: Logger instance
        max_attempts: Maximum number of attempts that were made
        final_error: Final exception that caused failure
        total_elapsed: Total time elapsed across all attempts
    """
    logger.error(
        f"All {max_attempts} retry attempts exhausted after {total_elapsed:.1f}s - "
        f"final error: {type(final_error).__name__}: {str(final_error)}"
    )


def log_retry_success(logger: logging.Logger, attempt: int, total_elapsed: float):
    """
    Log successful completion after retries.
    
    Args:
        logger: Logger instance
        attempt: Attempt number that succeeded (1-indexed)
        total_elapsed: Total time elapsed including retries
    """
    if attempt > 1:
        logger.info(
            f"Success on attempt {attempt} after {total_elapsed:.1f}s total elapsed"
        )