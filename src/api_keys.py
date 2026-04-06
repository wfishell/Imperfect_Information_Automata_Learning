"""
API Key Loader

Loads the appropriate API key for a given model name based on the provider
inferred from the model name prefix.

Keys are read from a .env file at the repo root (never committed to git).

.env format — one key=value per line:
    ANTHROPIC_API_KEY=sk-ant-...
    OPENAI_API_KEY=sk-...
    TOGETHER_API_KEY=...
    DEEPSEEK_API_KEY=...

Model routing:
    claude-*          → Anthropic
    gpt-*, o1, o3     → OpenAI
    deepseek-*        → DeepSeek direct API  (OpenAI-compatible)
    qwen-*            → Together AI
    llama-*           → Together AI
    together/*        → Together AI  (explicit namespace)
"""

import os

# Maps provider name → .env key name
_PROVIDER_ENV_KEY = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "together":  "TOGETHER_API_KEY",
    "deepseek":  "DEEPSEEK_API_KEY",
}

# Maps model name prefix → provider  (checked in order, first match wins)
_MODEL_PREFIX_TO_PROVIDER = [
    ("claude-",    "anthropic"),
    ("gpt-",       "openai"),
    ("o1",         "openai"),
    ("o3",         "openai"),
    ("deepseek-",  "deepseek"),   # direct DeepSeek API
    ("qwen-",      "together"),   # Qwen (Alibaba) via Together AI
    ("llama-",     "together"),   # Llama via Together AI
    ("together/",  "together"),   # explicit Together AI namespace
]

_REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_FILE   = os.path.join(_REPO_ROOT, ".env")


def _load_env_file() -> dict:
    """Parse .env file into a dict. Lines starting with # are ignored."""
    env = {}
    if not os.path.isfile(_ENV_FILE):
        return env
    with open(_ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env


def provider_for_model(model: str) -> str:
    """Infer the provider name from a model string."""
    lower = model.lower()
    for prefix, provider in _MODEL_PREFIX_TO_PROVIDER:
        if lower.startswith(prefix):
            return provider
    raise ValueError(
        f"Cannot infer provider for model '{model}'. "
        f"Known prefixes: {[p for p, _ in _MODEL_PREFIX_TO_PROVIDER]}"
    )


def load_api_key(model: str) -> str:
    """
    Return the API key for the provider that serves the given model.

    Lookup order:
      1. Environment variable (e.g. ANTHROPIC_API_KEY in the shell)
      2. .env file at the repo root

    Raises ValueError if the key is not found in either location.
    """
    provider = provider_for_model(model)
    env_var  = _PROVIDER_ENV_KEY[provider]

    # 1. Shell environment takes priority
    key = os.environ.get(env_var)
    if key:
        return key

    # 2. .env file
    file_env = _load_env_file()
    key = file_env.get(env_var)
    if key:
        return key

    raise ValueError(
        f"API key for provider '{provider}' not found. "
        f"Set {env_var} in your shell or in {_ENV_FILE}"
    )
