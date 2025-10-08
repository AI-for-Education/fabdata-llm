# %%
from pathlib import Path

from dotenv import load_dotenv
import tiktoken
import numpy as np

from fdllm import get_caller, LLMMessage
from fdllm.sysutils import register_models, list_models

load_dotenv(override=True)

register_models(Path.home() / ".fdllm/custom_models.yaml")
print(list_models())

# %%
caller = get_caller("gpt-4.1-mini")

# %%
msg = LLMMessage(Role="user", Message="Which is more red: A. cricket ball, B. apple. Answer only with the letter 'A' or 'B'")

# %%
enc = tiktoken.encoding_for_model(caller.Model.Api_Model_Name)
labels = ["A", "B"]
label_tokens = [enc.encode(label) for label in labels]
logit_bias = {
    str(token): 100 for token in set.union(*(set(tokens) for tokens in label_tokens))
}
max_tokens = max(len(tokens) for tokens in label_tokens)

# %%
out = caller.call(
    msg,
    max_tokens=max_tokens,
    logit_bias=logit_bias,
    logprobs=True,
    top_logprobs=20,
    top_p=1e-90,
)

lp = {
    lpt.token: lpt.logprob
    for lpt in out.LogProbs.content[0].top_logprobs
    if lpt.token in labels
}
lp = {lab: lp[lab] for lab in labels}
print(lp)

def logit_pairs_to_probs(logitsa, logitsb):
    def sigmoid(x):
        odds = np.exp(x)
        return odds / (odds + 1)

    logit_diffs = logitsa - logitsb
    probs = sigmoid(logit_diffs)
    out = np.zeros((probs.shape[0], 2))
    out[:, 0] = probs
    out[:, 1] = 1 - probs
    return out

print(logit_pairs_to_probs(*np.array(list(lp.values()))[:, None]).round(4))

