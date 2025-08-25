import os
from dotenv import load_dotenv

# Disables parallelism on tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# By default, load_dotenv searches for a .env file in the current directory 
# and then traverses up parent directories. When scripts are run from the 
# project root, it will find the .env file there.
load_dotenv()

def get_required_env(key):
    val = os.getenv(key)
    if val is None:
        raise ValueError(f"Error: The '{key}' environment variable must be set.")
    return val

def get_optional_env(key, default_val):
    return os.getenv(key, default_val)
