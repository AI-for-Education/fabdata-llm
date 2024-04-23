# %%
from pathlib import Path
from typing import List

from fdllm import get_caller
from fdllm.sysutils import register_models, list_models
from fdllm.chat import ChatController
from fdllm.tooluse import ToolUsePlugin, Tool, ToolParam


class TestTool(Tool):
    name = "mul"
    description = "Multiply 2 numbers"
    params = {
        "x": ToolParam(type="number", required=True),
        "y": ToolParam(type="number", required=True),
    }

    def execute(self, **params):
        res =  params["x"] * params["y"]
        return f"{res :.4f}"
    
    async def aexecute(self, **params):
        return self.execute()


class MyToolUsePlugin(ToolUsePlugin):
    Tools: List[Tool] = [TestTool()]


# %%
register_models(Path.home() / ".fdllm/custom_models.yaml")

# %%
chatter = ChatController(Caller=get_caller("gpt-4-1106-preview"))
plugin = MyToolUsePlugin()
chatter.register_plugin(plugin)

chatter.chat("pi x 5.4")
