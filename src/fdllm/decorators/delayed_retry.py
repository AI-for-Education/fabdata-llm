import time
from functools import wraps
import asyncio
import inspect
import logging
import threading
from typing import Optional

def pytest_exception_helper():
    return 1

def delayedretry(
    initial_wait=1,
    exponent=2,
    max_attempts=5,
    rethrow_final_error=False,
    include_errors=[],
    logger: Optional[logging.Logger] = None
):
    def mydec(func):
        # Thread-local storage to ensure each thread has its own retry state
        thread_local = threading.local()
        
        def get_retry_state():
            """Get or initialize retry state for current thread."""
            if not hasattr(thread_local, 'wait_time'):
                thread_local.wait_time = [initial_wait]
                thread_local.start_time = None
            return thread_local.wait_time, thread_local.start_time
        
        def set_start_time(start_time):
            """Set start time for current thread."""
            thread_local.start_time = start_time
        
        def reset_retry_state():
            """Reset retry state for current thread."""
            if hasattr(thread_local, 'wait_time'):
                thread_local.wait_time = [initial_wait]
                thread_local.start_time = None

        @wraps(func)
        async def awrapper(*args, **kwargs):
            wait_time, start_time = get_retry_state()
            
            # Initialize start time on first attempt
            if start_time is None:
                set_start_time(time.time())
                start_time = thread_local.start_time
            
            try:
                try:
                    out = await func(*args, **kwargs)
                    
                    # Log success if we had retries
                    if logger and len(wait_time) > 1:
                        total_elapsed = time.time() - start_time
                        logger.info(
                            f"Success on attempt {len(wait_time)} after {total_elapsed:.1f}s total elapsed"
                        )
                    
                    reset_retry_state()
                    return out
                except Exception as e:
                    attempt_num = len(wait_time)
                    total_elapsed = time.time() - start_time
                    
                    if (
                        attempt_num == max_attempts
                        or
                        not isinstance(e, tuple(include_errors))
                    ):
                        # Final failure or non-retryable error
                        if logger and attempt_num > 1:
                            logger.error(
                                f"All {max_attempts} retry attempts exhausted after {total_elapsed:.1f}s - "
                                f"final error: {type(e).__name__}: {str(e)}"
                            )
                        elif logger and not isinstance(e, tuple(include_errors)):
                            logger.debug(
                                f"Non-retryable error {type(e).__name__}: {str(e)}"
                            )
                        
                        reset_retry_state()
                        if rethrow_final_error:
                            raise
                        return None
                    
                    # Log retry attempt
                    if logger:
                        logger.warning(
                            f"Retry {attempt_num}/{max_attempts} - {type(e).__name__}: {str(e)} "
                            f"(waiting {wait_time[-1]:.1f}s, elapsed: {total_elapsed:.1f}s)"
                        )
                    
                    await asyncio.sleep(wait_time[-1])
                    pytest_exception_helper()
                    wait_time.append(wait_time[-1] * exponent)
                    return await awrapper(*args,**kwargs)
            except:
                reset_retry_state()
                raise
        @wraps(func)
        def wrapper(*args, **kwargs):
            wait_time, start_time = get_retry_state()
            
            # Initialize start time on first attempt
            if start_time is None:
                set_start_time(time.time())
                start_time = thread_local.start_time
            
            try:
                try:
                    out = func(*args, **kwargs)
                    
                    # Log success if we had retries
                    if logger and len(wait_time) > 1:
                        total_elapsed = time.time() - start_time
                        logger.info(
                            f"Success on attempt {len(wait_time)} after {total_elapsed:.1f}s total elapsed"
                        )
                    
                    reset_retry_state()
                    return out
                except Exception as e:
                    attempt_num = len(wait_time)
                    total_elapsed = time.time() - start_time
                    
                    if (
                        attempt_num == max_attempts
                        or
                        not isinstance(e, tuple(include_errors))
                    ):
                        # Final failure or non-retryable error
                        if logger and attempt_num > 1:
                            logger.error(
                                f"All {max_attempts} retry attempts exhausted after {total_elapsed:.1f}s - "
                                f"final error: {type(e).__name__}: {str(e)}"
                            )
                        elif logger and not isinstance(e, tuple(include_errors)):
                            logger.debug(
                                f"Non-retryable error {type(e).__name__}: {str(e)}"
                            )
                        
                        reset_retry_state()
                        if rethrow_final_error:
                            raise
                        return None
                    
                    # Log retry attempt
                    if logger:
                        logger.warning(
                            f"Retry {attempt_num}/{max_attempts} - {type(e).__name__}: {str(e)} "
                            f"(waiting {wait_time[-1]:.1f}s, elapsed: {total_elapsed:.1f}s)"
                        )
                    
                    time.sleep(wait_time[-1])
                    pytest_exception_helper()
                    wait_time.append(wait_time[-1] * exponent)
                    return wrapper(*args,**kwargs)
            except:
                reset_retry_state()
                raise
            
        isasync = inspect.iscoroutinefunction(func)
        if isasync:
            return awrapper
        else:
            return wrapper
    
    return mydec


# @myfactory()
# def myfunc(a):
#     rng = default_rng()
#     if rng.uniform() > a:
#         raise ValueError
#     else:
#         return a


# myfunc(0.1)
