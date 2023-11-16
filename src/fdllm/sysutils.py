from typing import Union
import os
from pathlib import Path
import yaml
from copy import deepcopy
from functools import reduce

HERE = Path(__file__).parent

CUSTOM_MODELS = [{}]

def list_models():
    models = load_models()
    return {key: list(val) for key, val in models.items()}

def register_models(modelfile: Union[str, os.PathLike]):
    modelfile = Path(modelfile)
    if not modelfile.exists():
        raise ValueError(f"{modelfile.as_posix()} doesn't exist")
    with open(modelfile, "r") as f:
        models = yaml.safe_load(f)
    CUSTOM_MODELS.append({})
    for key, val in models.items():
        CUSTOM_MODELS[-1][key] = val


def clear_model_register():
    while len(CUSTOM_MODELS) > 0:
        CUSTOM_MODELS.pop()


def load_models():
    basemodelfile = HERE / "models.yaml"
    with open(basemodelfile) as f:
        basemodels = yaml.safe_load(f)
    models = reduce(_deepmerge_models, [basemodels, *CUSTOM_MODELS])
    return models


def _deepmerge_models(model1, model2):
    if model1 is None:
        return model2
    out = deepcopy(model1)
    for key2, val2 in model2.items():
        if key2 in model1:
            val1 = model1[key2]
            if isinstance(val1, dict) and isinstance(val2, dict):
                out[key2] = _deepmerge_models(val1, val2)
            else:
                out[key2] = val2
        else:
            out[key2] = val2
    return out
