import json
import os
from copy import copy, deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Union

import yaml
from google.auth import default
from google.auth.transport import requests

HERE = Path(__file__).parent
basemodelfile = HERE / "models.yaml"


def deepmerge_dicts(dict1, dict2, copy_out=True):
    """Merge two dicts:

    - dict values are merged heirarchically
    - model2 keys overwrite model1 keys
    - does not change either of the two inputs, returns a deepcopy

    """
    if dict1 is None:
        return deepcopy(dict2)
    if copy_out:
        out = deepcopy(dict1)
    else:
        out = dict1

    for key2, val2 in dict2.items():
        if key2 in dict1:
            # use the copy so we don't have to deepcopy in the recursive call
            val1 = out[key2]
            if isinstance(val1, dict) and isinstance(val2, dict):
                out[key2] = deepmerge_dicts(val1, val2, copy_out=False)
            else:
                out[key2] = val2
        else:
            out[key2] = val2
    return out


def _parse_model_config_file(file: Union[str, os.PathLike]):
    """Load and parse the model definitions file"""
    file = Path(file)
    if not file.exists():
        raise ValueError(f"{file.as_posix()} doesn't exist")
    with open(file) as f:
        config = yaml.safe_load(f)

    # standalone model defintions
    models = config.pop("models", {})

    # grouped model definitions inherit group config
    for group_name, group_config in config.items():
        if "models" not in group_config:
            # no models defined for this group, raise error?
            continue
        group_models = group_config.pop("models")
        for model_name, model_config in group_models.items():
            # model keys take priority (second overwrites first)
            models[model_name] = deepmerge_dicts(group_config, model_config)
            models[model_name]["group"] = group_name

    return models


BASEMODELS = _parse_model_config_file(basemodelfile)
MODELS = copy(BASEMODELS)


def list_models(full_info=False, base_only=False):
    models = load_models(base_only)
    if full_info:
        return models
    else:
        return list(models)


def register_models(modelfile: Union[str, os.PathLike]):
    custom_models = _parse_model_config_file(modelfile)
    # latest file load overwrites previous
    clear_model_register()
    MODELS.update(deepmerge_dicts(BASEMODELS, custom_models))


def clear_model_register():
    MODELS.clear()


# change to get models
def load_models(base_only=False):
    if base_only:
        return BASEMODELS
    else:
        return MODELS


def get_google_token():
    def get_token():
        creds, _ = default()
        auth_req = requests.Request()
        creds.refresh(auth_req)
        return creds.token

    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        f = NamedTemporaryFile("w+t", delete=False)
        try:
            cred_json_str = os.environ.get("GOOGLE_AUTH_JSON").replace("\\n", "\n")
            cred_json = json.loads(cred_json_str)
            json.dump(cred_json, f)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name
            f.close()
            token = get_token()
            Path(f.name).unlink()
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS")
        except Exception as e:
            print(e)
            f.close()
            Path(f.name).unlink()
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS")
            token = None
    else:
        token = get_token()
    return token
