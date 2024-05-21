# %%
from pathlib import Path

import numpy as np
from fdllm import get_caller
from fdllm.sysutils import register_models
from fdllm.chat import ChatController
from fdllm.tooluse import ToolUsePlugin, Tool, ToolParam, ToolItem


class TestTool1(Tool):
    name = "mul"
    description = "Multiply 2 numbers"
    params = {
        "x": ToolParam(type="number", required=True),
        "y": ToolParam(type="number", required=True),
    }

    def execute(self, **params):
        res = params["x"] * params["y"]
        return f"{res :.4f}"

    async def aexecute(self, **params):
        return self.execute()


class TestTool2(Tool):
    name = "add"
    description = "Add 2 numbers"
    params = {
        "x": ToolParam(type="number", required=True),
        "y": ToolParam(type="number", required=True),
    }

    def execute(self, **params):
        res = params["x"] + params["y"]
        return f"{res :.4f}"

    async def aexecute(self, **params):
        return self.execute()


class TestTool3(Tool):
    name = "dot_product"
    description = (
        "Calculate the dot product between 2 arrays of numbers."
        " The dot product is the sum of the products of the individual"
        " elements of each array."
    )
    params = {
        "x": ToolParam(type="array", items=ToolItem(type="number"), required=True),
        "y": ToolParam(type="array", items=ToolItem(type="number"), required=True),
    }

    def execute(self, **params):
        res = np.dot(params["x"], params["y"])
        return f"{res :.4f}"

    async def aexecute(self, **params):
        return self.execute()


# %%
register_models(Path.home() / ".fdllm/custom_models.yaml")

# %%
model = "gpt-4-1106-preview"
chatter = ChatController(
    Caller=get_caller(model),
    Sys_Msg={0: "Use tools for all calculations where possible."},
)
chatter.register_plugin(ToolUsePlugin(Tools=[TestTool1(), TestTool2(), TestTool3()]))

# %%
## This example is possible with a single tool call
prompt = "pi x 5.4"

chatter.chat(prompt)
chatter.History

# %%
## This example should require 3 tool calls to calculate
## The 2 additions could be computed in parallel tool calls,
## the final multiplication is contingent on the outputs of the 2 additions
prompt = "(pi + 5.4) * (6 + e)"

# clear chat history
chatter.History = []

chatter.chat(prompt)
chatter.History

# %%
## This example could use 3 tool calls to calculate:
## (the 2 multiplications followed by the addition),
## or it could be done in one call to the 'dot_product' tool instead.
## (Seems like that strategy is never used by gpt-4-1106-preview)
prompt = "(pi * 5.4) + (6 * e)"

# clear chat history
chatter.History = []

chatter.chat(prompt)
chatter.History
