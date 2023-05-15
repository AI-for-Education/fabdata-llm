import pytest
from unittest.mock import patch, MagicMock

from fdllm.decorators.delayed_retry import delayedretry

@pytest.mark.parametrize(
    'num_times_to_fail, expected_response',
    [
        (0, "Success"),
        (1, "Success"),
        (5, None)
    ]
)
def test_sync_decorator(num_times_to_fail, expected_response):
    max_attempts = 3
    def test_func_wrapper():
        num_attempts = 0

        @delayedretry(initial_wait=1, exponent=1, max_attempts=max_attempts)
        def test_func():
            nonlocal num_attempts
            num_attempts = num_attempts + 1
            if num_attempts <= num_times_to_fail:
                raise ValueError("max Attempts reached")
            
            return (num_attempts, expected_response)
            
        return test_func
    
    with patch("time.sleep", return_value = None):
        response = test_func_wrapper()()

    if response:
        (num_attempts, actual_response) = response

        assert num_attempts == num_times_to_fail + 1
        assert actual_response == expected_response
    else:
        assert response == expected_response
    

def test_sync_decorator_waitexception():
    max_attempts = 3
    num_times_to_fail = 1
    expected_response = "Test exception"

    class DemoException(Exception):
        pass

    def test_func_wrapper():
        num_attempts = 0

        @delayedretry(initial_wait=1, exponent=1, max_attempts=max_attempts)
        def test_func():
            nonlocal num_attempts
            num_attempts = num_attempts + 1
            if num_attempts <= num_times_to_fail:
                raise ValueError("max Attempts reached")
            
            return (num_attempts, expected_response)
            
        return test_func
    
    with patch("time.sleep", return_value = None):
        with patch("llm.decorators.delayed_retry.pytest_exception_helper") as patch_fn:
            patch_fn.side_effect = DemoException(expected_response)
            with pytest.raises(DemoException) as err_info:
                response = test_func_wrapper()()

    assert str(err_info.value) == expected_response
    assert patch_fn.call_count == 1
    


@pytest.mark.parametrize(
    'num_times_to_fail, expected_response',
    [
        (0, "Success"),
        (1, "Success"),
        (5, None)
    ]
)
async def test_async_decorator(anyio_backend, num_times_to_fail, expected_response):
    max_attempts = 3
    async def test_func_wrapper_async():
        num_attempts = 0

        @delayedretry(initial_wait=1, exponent=1, max_attempts=max_attempts)
        async def test_func_async():
            nonlocal num_attempts
            num_attempts = num_attempts + 1
            if num_attempts <= num_times_to_fail:
                raise ValueError("max Attempts reached")
            
            return (num_attempts, expected_response)
            
        return test_func_async
    
    wrapper = await test_func_wrapper_async()
    with patch("asyncio.sleep", return_value = None):
        response = await wrapper()
    if response:
        (num_attempts, actual_response) = response

        assert num_attempts == num_times_to_fail + 1
        assert actual_response == expected_response
    else:
        assert response == expected_response


async def test_async_decorator_waitexception(anyio_backend):
    max_attempts = 3
    num_times_to_fail = 1
    expected_response = "Test exception"

    class DemoException(Exception):
        pass

    async def test_func_wrapper_async():
        num_attempts = 0

        @delayedretry(initial_wait=1, exponent=1, max_attempts=max_attempts)
        async def test_func_async():
            nonlocal num_attempts
            num_attempts = num_attempts + 1
            if num_attempts <= num_times_to_fail:
                raise ValueError("max Attempts reached")
            
            return (num_attempts, expected_response)
            
        return test_func_async
    
    wrapper = await test_func_wrapper_async()
    with patch("asyncio.sleep", return_value = None):
        with patch("llm.decorators.delayed_retry.pytest_exception_helper") as patch_fn:
            patch_fn.side_effect = DemoException(expected_response)
            with pytest.raises(DemoException) as err_info:
                response = await wrapper()
    
    assert str(err_info.value) == expected_response
    assert patch_fn.call_count == 1
