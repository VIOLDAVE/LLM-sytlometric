"""
Test whether API keys are being loaded from the .env file.

This script does not print full keys.
It only confirms whether each key exists.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

print(f"Looking for .env file at: {ENV_PATH}")

if not ENV_PATH.exists():
    raise FileNotFoundError(f".env file not found at: {ENV_PATH}")

load_dotenv(dotenv_path=ENV_PATH, override=True)

api_keys = {
    "OpenAI": "OPENAI_API_KEY",
    "Claude": "ANTHROPIC_API_KEY",
    "Gemini": "GOOGLE_API_KEY",
    "DeepSeek": "DEEPSEEK_API_KEY",
    "Mistral": "MISTRAL_API_KEY",
}

for provider, env_name in api_keys.items():
    value = os.getenv(env_name)

    if value:
        hidden_key = value[:6] + "..." + value[-4:]
        print(f"{provider}: key found ✅ {hidden_key}")
    else:
        print(f"{provider}: key missing ❌ Expected variable name: {env_name}")