import os
import sys
import threading
from dotenv import load_dotenv

# Windows stdout/stderr default to cp1252 when piped/redirected, which can crash with
# UnicodeEncodeError on non-ASCII characters in log output. Force UTF-8 here, before any
# other module is imported, so redirected runs never crash. errors="replace" guards
# against any remaining exotic character.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

load_dotenv()

# Jetson Vision LLM (Gemma E2B - camera, routing calls, fast tasks)
LLAMA_URL = os.getenv("LLAMA_URL", "")
MODEL = os.getenv("MODEL", "unsloth/gemma-4-E2B-it-GGUF:Q4_K_M")
VISION_LLM_URL = os.getenv("VISION_LLM_URL", LLAMA_URL)
VISION_LLM_MODEL = os.getenv("VISION_LLM_MODEL", MODEL)

# Laptop Reasoning LLM (Qwen3-8B via Ollama, GPU - heavy debate agents + oracle)
REASONING_LLM_URL = os.getenv("REASONING_LLM_URL", "http://localhost:11434/v1/chat/completions")
REASONING_LLM_MODEL = os.getenv("REASONING_LLM_MODEL", "qwen3:8b")

# Laptop Diversity LLM (Llama 3.1 8B via Ollama, GPU). A different model family
# (distinct training/biases from Qwen + Gemma) - the cognitive diversity that makes
# debate beat a single model. Ollama serializes Qwen/Llama swaps (~8GB total > 8GB VRAM)
# but GPU inference (~4-8s/call) is far better than the old CPU-pinned 120s timeout.
DIVERSITY_LLM_URL = os.getenv("DIVERSITY_LLM_URL", REASONING_LLM_URL)
DIVERSITY_LLM_MODEL = os.getenv("DIVERSITY_LLM_MODEL", "llama3.1:8b")

LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "600"))  # kept for backward compat

# Safety
# Hard cap per single LLM call - on timeout we abort, no retry
# 120s: Qwen3 thinking calls (num_predict=2048) can run 100-110s; Gemma tops out ~30s
MAX_LLM_CALL_SECONDS = int(os.getenv("MAX_LLM_CALL_SECONDS", "120"))
# Retry on transient 5xx / connection errors (never on timeout or 4xx)
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "2.0"))
# Circuit breaker: open after N consecutive failures, reset after M seconds
CB_FAILURE_THRESHOLD = int(os.getenv("CB_FAILURE_THRESHOLD", "5"))
CB_RESET_SECONDS = int(os.getenv("CB_RESET_SECONDS", "60"))

# Neo4j - try local first, fall back to Jetson
LOCAL_NEO4J_URI = os.getenv("LOCAL_NEO4J_URI", "bolt://localhost:7687")
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "2"))  # >=2 activates DART, SELENE, MoA, belief-calib
AGENTS_PER_ROUND = int(os.getenv("AGENTS_PER_ROUND", "5"))

# Dynamic runtime LLM config (set via frontend gear icon)
_config_lock = threading.Lock()

_PROVIDER_URLS: dict[str, str] = {
    "local": LLAMA_URL,
    "openai": "https://api.openai.com/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
}

_PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "local": MODEL,
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
}

_llm_config: dict = {
    "provider": "local",
    "api_key": "",
    "base_url": LLAMA_URL,
    "model": MODEL,
}


def get_llm_config() -> dict:
    with _config_lock:
        return _llm_config.copy()


def get_vision_llm_config() -> dict:
    """Jetson Gemma E2B - always used for camera snapshots, routing calls, fast tasks."""
    return {
        "provider": "local",
        "api_key": "",
        "base_url": VISION_LLM_URL,
        "model": VISION_LLM_MODEL,
        "supports_thinking_param": True,  # llama-server supports enable_thinking
    }


def get_reasoning_llm_config() -> dict:
    """
    Laptop Qwen3-14B (Ollama) - used for swarm debate agents and oracle synthesis.
    If the user has overridden to a cloud provider via the frontend gear settings,
    that provider is used for reasoning calls instead.
    """
    with _config_lock:
        cfg = _llm_config.copy()
    if cfg["provider"] != "local":
        cfg.setdefault("supports_thinking_param", False)
        return cfg
    return {
        "provider": "local",
        "api_key": "",
        "base_url": REASONING_LLM_URL,
        "model": REASONING_LLM_MODEL,
        "supports_thinking_param": False,  # Ollama ignores enable_thinking; we strip tags
    }


def get_diversity_llm_config() -> dict:
    """
    Llama 3.1 8B (Ollama, CPU-pinned) - the third model family, for cognitive diversity.
    Different training and blind spots from Qwen and Gemma, so it catches errors they
    share. Runs on CPU so it never contends with Qwen for the 8 GB GPU.
    """
    return {
        "provider": "local",
        "api_key": "",
        "base_url": DIVERSITY_LLM_URL,
        "model": DIVERSITY_LLM_MODEL,
        "supports_thinking_param": False,
    }


def set_llm_config(provider: str, api_key: str = "", model: str = "", base_url: str = "") -> None:
    with _config_lock:
        _llm_config["provider"] = provider
        _llm_config["api_key"] = api_key
        _llm_config["base_url"] = base_url or _PROVIDER_URLS.get(provider, LLAMA_URL)
        _llm_config["model"] = model or _PROVIDER_DEFAULT_MODELS.get(provider, MODEL)
