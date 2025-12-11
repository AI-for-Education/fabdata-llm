import pytest
from pathlib import Path

from dotenv import load_dotenv

import numpy as np
from PIL import Image
from fdllm.llmtypes import LLMMessage, LLMImage

HERE = Path(__file__).resolve().parent
TEST_ROOT = HERE

load_dotenv(TEST_ROOT / "test.env", override=True)

TEST_MESSAGE = LLMMessage(Role="user", Message="Test input")
TEST_IM_ARR = np.zeros((2048, 4096, 3))
TEST_IM = Image.fromarray(((TEST_IM_ARR) * 255).astype(np.uint8))


@pytest.mark.parametrize(
    "detail, expected", [("low", (512, 512)), ("high", (1536, 768))]
)
def test_llm_image(detail, expected):
    im = LLMImage(Img=TEST_IM, Detail=detail)
    assert im.Img.size == expected
