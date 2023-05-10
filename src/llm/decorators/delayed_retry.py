import time
from functools import wraps
import asyncio
import inspect

def pytest_exception_helper():
    return 1

def delayedretry(initial_wait=1, exponent=2, max_attempts=5, rethrow_final_error = False):
    def mydec(func):
        wait_time = [initial_wait]
        def reset_wait_time():
            while len(wait_time) > 1:
                wait_time.pop(-1)

        @wraps(func)
        async def awrapper(*args, **kwargs):
            # print(f"async count: {len(wait_time)}")
            try:
                try:
                    out = await func(*args, **kwargs)
                    reset_wait_time()
                    return out
                except:
                    if len(wait_time) == max_attempts:
                        reset_wait_time()
                        if rethrow_final_error:
                            raise
                        return None
                    
                    await asyncio.sleep(wait_time[-1])
                    pytest_exception_helper()
                    wait_time.append(wait_time[-1] * exponent)
                    return await awrapper(*args,**kwargs)
            except:
                reset_wait_time()
                raise
        @wraps(func)
        def wrapper(*args, **kwargs):
            # print(f"count: {len(wait_time)}")
            try:
                try:
                    out = func(*args, **kwargs)
                    reset_wait_time()
                    return out
                except:
                    if len(wait_time) == max_attempts:
                        reset_wait_time()
                        if rethrow_final_error:
                            raise
                        return None
                    
                    time.sleep(wait_time[-1] * exponent)
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
