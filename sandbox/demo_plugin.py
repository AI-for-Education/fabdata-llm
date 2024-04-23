# %%
from pathlib import Path
from typing import Optional, List, Dict
from collections import Counter
import json

from pydantic import Field
from fdllm import get_caller
from fdllm.chat import ChatController, ChatPlugin
from fdllm.llmtypes import LLMMessage, LLMCaller
from fdllm.sysutils import register_models, list_models

modelfile = Path.home() / ".fdllm/custom_models.yaml"
if modelfile.exists():
    register_models(modelfile)
print(list_models())


# %%
class CopycatPlugin(ChatPlugin):
    """
    Keeps a running tally of the user's most frequent words and injects an
    instruction into the system message to use these words in the responses.
    
    Not a very useful idea in itself, but demonstrates some features of ChatPlugin.
    After being registered, plugin will have reference to parent ChatController
    in self.Controller
    """

    ### base fields from ChatPlugin
    # This is a separate Caller that the plugin can use for whatever reason
    # Need to be declared like this because in parent class it is not optional.
    # That fact will likely change in future.
    Caller: Optional[LLMCaller] = None
    # These attributes of ChatController will be restored to their previous
    # state after the plugin finishes
    Restore_Attrs: List[str] = ["Sys_Msg"]

    ### unique fields
    word_count: Dict[str, Counter] = Field(
        default_factory=lambda: {"user": Counter(), "assistant": Counter()}
    )
    n: int = 5
    verbose: bool = False

    #####################
    #### base methods from ChatPlugin ABC
    def register(self):
        """
        This is called when the the plugin is registered with
        ChatController.register_plugin
        """
        # reset word counter
        for key in self.word_count:
            self.word_count[key] = Counter()

    def unregister(self):
        """
        This is called when the the plugin is unregistered with
        ChatController.unregister_plugin
        """
        return super().unregister()

    def pre_chat(self, prompt: str, *args, **kwargs):
        """
        This is called on the first line of ChatController.chat,
        in ChatController._run_plugins, prior to any other effects 
        of the prompt on the state of the ChatController object.
        Multiple plugins are run in order that they were registered
        """
        # increment user word counter
        self.update_count("user", prompt)

        # Add some silly instruction to the system message
        # Because Sys_Msg is in Restore_Attrs, it will be restored back to its
        # state prior to the plugin run after the plugin has finished
        fav_words = [mc[0] for mc in self.word_count["user"].most_common(self.n)]
        smsg_appender = (
            f"\nHere is a list of the user's {self.n} favourite words:"
            f"\n{', '.join(fav_words)}"
            "\nUse them all in your response."
        )
        Sys_Msg = self.Controller.Sys_Msg
        ### Sys_Msg[-1] is appended to the end of the input sequence so it tends to be
        ### more salient than Sys_Msg[0], which appears at the start
        Sys_Msg[-1] = Sys_Msg.get(-1, "") + smsg_appender

        if self.verbose:
            print(self.Controller.Sys_Msg)

    async def pre_achat(self, prompt: str, *args, **kwargs):
        """
        Here we just call and return the sync version
        """
        return self.pre_chat(prompt, *args, **kwargs)

    def post_chat(self, result: LLMMessage, *args, **kwargs):
        """
        This is called on the last line of ChatController.chat, 
        in ChatController._clean_plugins, and is the last thing that 
        can have any effect on the state of the ChatController object.
        Multiple plugins are cleaned in the reverse order that they were 
        registered.
        """
        # increment assistant word counter
        # This count isn't actually used but just demonstrates something that needs
        # to take place after the response has come back
        if result.Message is not None:
            self.update_count("assistant", result.Message)

        return result

    async def post_achat(self, result: LLMMessage, *args, **kwargs):
        """
        Here we just call and return the sync version
        """
        return self.post_chat(result, *args, **kwargs)

    ####################
    ### unique methods

    def update_count(self, role, string: str):
        for word in string.split():
            # here you could remove stop words but to save adding a new
            # dependency to the package just filter out short words
            if len(word) > 3:
                self.word_count[role][word] += 1


# %%
chat = ChatController(Caller=get_caller("gpt-3.5-turbo"))

plugin = CopycatPlugin()

chat.register_plugin(plugin)

# %%
input, output = chat.chat("That is truly, truly, truly amazing")
print(output.Message)

# check that our controller's default Sys_Msg is restored after the plugin has
# run
print(chat.Sys_Msg)
# %%
input, output = chat.chat("Can you help me with maths")
print(output.Message)

# %%
input, output = chat.chat("Algebra")
print(output.Message)
