# %%
#
# Example script that runs through all the functionality for a model
#

# ****************************

model = "gemini-2.0-flash-exp"

# ****************************

# %%
import time
from functools import wraps
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from fdllm import get_caller, register_models, LLMMessage, ChatController
from fdllm.tooluse import ToolUsePlugin, Tool, ToolParam

load_dotenv(override=True)
# register_models("../custom_models.yaml")


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"... {total_time:.4f} seconds")
        return result

    return timeit_wrapper


print(f"fdllm demo for {model}...")

# %%
caller = get_caller(model)
has_vision = caller.Model.Vision
has_tools = caller.Model.Tool_Use
sep = "\n----------------------------------\n"
# %%
print(sep)
print("Testing caller...")
prompt = "Hi there"


@timeit
def run_caller(caller, prompt):
    print(f"\nUser: {prompt}")
    print("Model: " + caller.call(LLMMessage(Role="user", Message=prompt)).Message)


run_caller(caller, prompt)
print(sep)
# %%
print(sep)
print("Testing chatter...")
chatter = ChatController(Caller=caller)


@timeit
def run_chatter(chatter, prompts):
    for p in prompts:
        print(f"\nUser: {p}")
        inmsg, outmsg = chatter.chat(p)
        print("Model: " + outmsg.Message)


prompts = ["Hi there", "Can you help me with maths", "Algebra"]
run_chatter(chatter, prompts)
print(sep)


# %%
@timeit
def run_vision(chatter, images):
    chatter.History = []
    prompt = "Compare these 2 paintings"
    print(f"\nUser: {prompt}")
    inmsg, outmsg = chatter.chat(prompt, images=images, detail="high")
    print("Model: " + outmsg.Message)
    prompt = "Can you say more?"
    print(f"\nUser: {prompt}")
    inmsg, outmsg = chatter.chat(prompt)
    print("Model: " + outmsg.Message)


if has_vision:
    ROOT = Path(__file__).parents[1]
    testims = [Image.open(f) for f in (ROOT / "assets").glob("*")]
    print(sep)
    print("Testing vision...")
    run_vision(chatter, images=testims)
    print(sep)


# %%

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


@timeit
def run_tools(chatter):
    chatter.History = []
    prompt = "what is 3.14 x 5.47"
    inmsg, outmsg = chatter.chat(prompt)

    print(f"\nUser: {prompt}")
    print("Tool calls: ")
    print("    " + "\n    ".join([repr(t) for t in chatter.History[1].ToolCalls]))
    print("Model: " + outmsg.Message)

    chatter.History = []
    prompt = "(3.14 + 5.47) * (6 + 2.71)"
    inmsg, outmsg = chatter.chat(prompt)

    print(f"\nUser: {prompt}")
    print("Tool calls: ")
    tcs = [msg.ToolCalls for msg in chatter.History if msg.ToolCalls and msg.Role=="assistant"]
    for tc in tcs:
        print("    " + "\n    ".join([repr(t) for t in tc]))
    print("Model: " + outmsg.Message)


#%%
if has_tools:
    print(sep)
    print("Testing tool use...")
    chatter.Sys_Msg = {0: "Use tools for all calculations where possible."}
    chatter.register_plugin(ToolUsePlugin(Tools=[TestTool1(), TestTool2()]))

    prompt = "what is 3.14 x 5.47"
    inmsg, outmsg = chatter.chat(prompt)
    run_tools(chatter)

