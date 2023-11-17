# %%
from pathlib import Path

from PIL import Image
from fdllm import get_caller
from fdllm.sysutils import register_models, list_models
from fdllm.llmtypes import LLMMessage, LLMImage
from fdllm.chat import ChatController

register_models(Path(__file__).parents[1] / "custom_models.yaml")
print(list_models(full_info=True, base_only=True))

ROOT = Path(__file__).parents[1]

testims = [Image.open(f) for f in (ROOT / "assets").glob("*")]

# %%
caller = get_caller("gpt-4-vision-preview")

# %%
message = LLMMessage(
    Role="user",
    Message="Compare these 2 paintings",
    Images=LLMImage.list_from_images(testims, detail="high")
)

print(caller.call([message]))

# %%
chatter = ChatController(Caller=caller)

# %%
chatter.chat("Compare these 2 paintings", images=testims, detail="high")

# %%
chatter.chat("Can you say more")