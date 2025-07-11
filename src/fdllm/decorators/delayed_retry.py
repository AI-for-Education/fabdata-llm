import time
from functools import wraps
import asyncio
import inspect
import logging
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
        wait_time = [initial_wait]
        start_time = [None]  # Track start time for total elapsed calculation
        
        def reset_wait_time():
            while len(wait_time) > 1:
                wait_time.pop(-1)
            start_time[0] = None

        @wraps(func)
        async def awrapper(*args, **kwargs):
            # Initialize start time on first attempt
            if start_time[0] is None:
                start_time[0] = time.time()
            
            try:
                try:
                    out = await func(*args, **kwargs)
                    
                    # Log success if we had retries
                    if logger and len(wait_time) > 1:
                        total_elapsed = time.time() - start_time[0]
                        logger.info(
                            f"Success on attempt {len(wait_time)} after {total_elapsed:.1f}s total elapsed"
                        )
                    
                    reset_wait_time()
                    return out
                except Exception as e:
                    attempt_num = len(wait_time)
                    total_elapsed = time.time() - start_time[0]
                    
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
                        
                        reset_wait_time()
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
                reset_wait_time()
                raise
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Initialize start time on first attempt
            if start_time[0] is None:
                start_time[0] = time.time()
            
            try:
                try:
                    out = func(*args, **kwargs)
                    
                    # Log success if we had retries
                    if logger and len(wait_time) > 1:
                        total_elapsed = time.time() - start_time[0]
                        logger.info(
                            f"Success on attempt {len(wait_time)} after {total_elapsed:.1f}s total elapsed"
                        )
                    
                    reset_wait_time()
                    return out
                except Exception as e:
                    attempt_num = len(wait_time)
                    total_elapsed = time.time() - start_time[0]
                    
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
                        
                        reset_wait_time()
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
                reset_wait_time()
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
