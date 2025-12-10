import json

from . import get_caller
from .llmtypes import LLMMessage, LLMImage


class ADict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, val in self.items():
            if isinstance(val, dict):
                self[key] = self.__class__(val)
            elif isinstance(val, list):
                self[key] = [
                    self.__class__(val_) if isinstance(val_, dict) else val_
                    for val_ in val
                ]

    def __getitem__(self, key):
        for k, val in self.items():
            if key in [k.split("::")[0], k]:
                return val
        raise IndexError()

    def __setitem__(self, key, value):
        for k in self:
            if key in [k.split("::")[0], k]:
                super().__setitem__(k, value)
                return
        super().__setitem__(key, value)

    def to_dict(self):
        return _clean_keys(dict(self))


def general_query(
    jsonin,
    jsonout,
    caller=None,
    role="system",
    temperature=0,
    max_input_tokens=None,
    min_new_token_window=500,
    reduce_callback=None,
    images=[],
    detail="low",
    **call_kwargs,
):
    if caller is None:
        caller = get_caller("gpt-4")
    if images and role == "system":
        raise ValueError("Can''t provide images if role=='system'")
    msg = _gen_message(jsonin, jsonout, role, images, detail)
    ntok = caller.count_tokens([msg])
    if max_input_tokens is not None and ntok > max_input_tokens:
        raise ValueError("Message is too long")
    max_tokens = caller.token_window - ntok
    if max_tokens < min_new_token_window:
        if reduce_callback is None:
            raise ValueError("Message is too long")
        else:
            while max_tokens < min_new_token_window:
                jsonin, jsonout = reduce_callback(jsonin, jsonout)
                msg = _gen_message(jsonin, jsonout, images)
                ntok = caller.count_tokens([msg])
                max_tokens = caller.token_window - ntok
    if temperature is not None:
        call_kwargs = {**call_kwargs, "temperature": temperature}
    out = caller.call(msg, max_tokens=max_tokens, **call_kwargs)

    try:
        return ADict(json.loads(_trim_nonjson(out.message)))
    except:
        raise ValueError("Invalid output")


def _gen_message(jsonin, jsonout, role="system", images=[], detail="low"):
    return LLMMessage(
        role=role,
        message=(
            "Given the values in JSON1, fill in the empty values in JSON2:"
            f"\n\nJSON1:\n{json.dumps(jsonin, ensure_ascii=False)}"
            f"\n\nJSON2:\n{json.dumps(jsonout, ensure_ascii=False)}"
            "\n\nExpand any lists where necessary. Only return the raw json."
            " For any field names that contain '::', only reproduce the part of the"
            " name before the '::'."
        ),
        images=LLMImage.list_from_images(images, detail=detail) if images else None,
    )


def _clean_keys(d):
    if not isinstance(d, dict):
        return d
    out = d.copy()
    for key in d:
        usekey = key.split("::")[0]
        useval = out.pop(key)
        if isinstance(useval, dict):
            out[usekey] = _clean_keys(useval)
        elif isinstance(useval, list):
            out[usekey] = [_clean_keys(uv) for uv in useval]
        else:
            out[usekey] = useval
    return out


def _trim_nonjson(text):
    pre, *post = text.split("{")
    text = "{".join(["", *post])
    *pre, post = text.split("}")
    text = "}".join([*pre, ""])
    return text
