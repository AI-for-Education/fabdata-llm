import pytest

import numpy as np
from PIL import Image
from fdllm.llmtypes import LLMMessage, LLMImage

TEST_MESSAGE = LLMMessage(Role="user", Message="Test input")
TEST_IM_ARR_LARGE = np.zeros((2048, 4096, 3))
TEST_IM_ARR_SMALL = np.zeros((200, 300, 3))
TEST_IM_LARGE = Image.fromarray(((TEST_IM_ARR_LARGE) * 255).astype(np.uint8))
TEST_IM_SMALL = Image.fromarray(((TEST_IM_ARR_SMALL) * 255).astype(np.uint8))


@pytest.mark.parametrize(
    "image, detail, expected",
    [
        (TEST_IM_LARGE, "low", (512, 256)),
        (TEST_IM_LARGE, "high", (1536, 768)),
        (TEST_IM_SMALL, "low", (300, 200)),
        (TEST_IM_SMALL, "high", (300, 200)),
    ],
)
def test_llm_image(image, detail, expected):
    im = LLMImage(Img=image, Detail=detail)
    assert im.Img.size == expected
