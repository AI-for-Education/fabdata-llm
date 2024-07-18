# %%
from pathlib import Path
import os
from tempfile import NamedTemporaryFile
import time
import json

from dotenv import load_dotenv
load_dotenv(override=True)

from google.auth import default, transport

def get_google_token():
    def get_token():
        creds, _ = default()
        auth_req = transport.requests.Request()
        creds.refresh(auth_req)
        return creds.token
        
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        f = NamedTemporaryFile("w+t", delete=False)
        try:
            cred_json_str = os.environ.get("GOOGLE_AUTH_JSON")
            cred_json = json.loads(cred_json_str)
            json.dump(cred_json, f)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name
            f.close()
            token = get_token()
            Path(f.name).unlink()
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS")
        except Exception:
            Path(f.name).unlink()
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS")
            token = None
    else:
        token = get_token()
    return token


import vertexai
from vertexai.generative_models import GenerativeModel

project_id = "bionic-run-429809-p5"

# %%
vertexai.init(project=project_id, location="us-central1")

model = GenerativeModel("gemini-1.5-flash-001")

# %%
response = model.generate_content(
    "What's a good name for a flower shop that specializes in selling bouquets of dried flowers?"
)

print(response.text)
