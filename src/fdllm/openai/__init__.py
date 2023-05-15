from os import getenv

import openai

openai_key = str(getenv('OPENAI_KEY'))
openai.api_key = openai_key

from .caller import GPTCaller
