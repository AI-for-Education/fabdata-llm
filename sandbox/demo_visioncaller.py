# %%
from pathlib import Path

from PIL import Image
from fdllm import GPTVisionCaller
from fdllm.llmtypes import LLMMessage, LLMImage
from fdllm.chat import ChatController

ROOT = Path(__file__).parents[1]

testims = [Image.open(f) for f in (ROOT / "assets").glob("*")]

# %%
caller = GPTVisionCaller("gpt-4-vision-preview")

# %%
message = LLMMessage(
    Role="user",
    Message="Compare these 2 paintings",
    Images=LLMImage.list_from_images(testims, detail="low")
)

print(caller.call([message]))

# %%
chatter = ChatController(Caller=caller)

# %%
chatter.chat("Compare these 2 paintings", images=testims)

# %%
chatter.chat("Can you say more")