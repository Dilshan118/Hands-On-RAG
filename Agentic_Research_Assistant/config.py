"""
config.py — Multi-Provider LLM Factory
=======================================

📚 BEGINNER EXPLANATION:
    This file is the **single point of control** for creating LLM (Large Language Model)
    instances across the entire Agentic Research Assistant system. Instead of every agent
    file (planner.py, writer.py, critic.py) independently importing and configuring
    their own LLM objects, they all call ONE function: `get_llm()`.

🏗️ DESIGN PATTERN — Factory Method:
    The Factory Method pattern is a creational design pattern that provides a single
    interface for creating objects without exposing the creation logic to the caller.
    
    WHY USE IT HERE?
    - Agent nodes should NOT care which LLM provider (Groq, Gemini, Ollama) is active.
    - Switching from Groq → Gemini requires changing ONE environment variable, not
      editing every agent file.
    - This is called "Loose Coupling" — components depend on abstractions, not implementations.

    ANALOGY: Think of this as a universal power adapter. Your laptop (agent) doesn't care
    if the wall socket is US, UK, or EU — the adapter (factory) handles the conversion.

🔧 SUPPORTED PROVIDERS:
    1. Groq   — Cloud LPU inference (recommended free tier, ultra-fast)
    2. Google  — Gemini API (free tier via AI Studio)
    3. Ollama  — 100% local offline inference (no API key needed)

📁 ARCHITECTURE ROLE:
    This file sits at LAYER 5 (Infrastructure) of the system architecture.
    Every agent node in LAYER 3 calls `get_llm()` without knowing which provider is active.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

# ──────────────────────────────────────────────────────────────
# Setup logging so we can trace which provider is being used.
# In production systems, logging replaces print() statements
# because logs can be routed to files, dashboards, or monitoring tools.
# ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Load environment variables from the .env file.
# python-dotenv reads key=value pairs from .env and injects them
# into os.environ so our code can access API keys securely.
#
# WHY .env FILES?
#   - Never hardcode API keys in source code (security risk)
#   - .env files are excluded from Git via .gitignore
#   - Different environments (dev/staging/prod) use different .env files
# ──────────────────────────────────────────────────────────────
load_dotenv()

# ──────────────────────────────────────────────────────────────
# Global Configuration Constants
# These are the "sensible defaults" for the system. They can be
# overridden via environment variables or the Streamlit sidebar.
# ──────────────────────────────────────────────────────────────
DEFAULT_PROVIDER = "groq"                    # Which LLM provider to use by default
DEFAULT_MODEL_NAME = "llama-3.3-70b-versatile"  # Default Groq model (70B parameter Llama 3.3)
EMBEDDING_MODEL_NAME = "models/text-embedding-004"  # Google embedding model for vector search
MAX_REVISIONS = 2                            # Maximum reflection loops before forced finalization
CHROMA_PERSIST_DIR = os.path.join(           # Local persistent storage path for ChromaDB vectors
    os.path.dirname(__file__), "chroma_db"
)


def get_llm(model_name: str = None, temperature: float = 0.2) -> BaseChatModel:
    """
    Factory function that creates and returns the appropriate LLM instance
    based on the configured provider (Groq, Google Gemini, or Ollama).

    📚 BEGINNER NOTE — What is "temperature"?
        Temperature controls the randomness/creativity of LLM outputs.
        - temperature=0.0 → Deterministic, always picks the most likely token
        - temperature=0.2 → Slightly creative but mostly factual (our default)
        - temperature=1.0 → Highly creative/random, good for brainstorming
        
        For a research assistant that needs ACCURATE facts, we keep temperature LOW.

    📚 BEGINNER NOTE — What is BaseChatModel?
        BaseChatModel is an abstract base class from LangChain. It defines the
        interface that ALL chat models must implement (e.g., `.invoke(prompt)`).
        By returning BaseChatModel, our agents don't need to know if they're
        talking to Groq, Gemini, or Ollama — they just call `.invoke()`.
        This is called "Programming to an Interface" (a SOLID principle).

    Args:
        model_name: Optional override for the model name. If None, uses the
                    provider-specific default from environment variables.
        temperature: Controls output randomness. Lower = more deterministic.

    Returns:
        BaseChatModel: A configured LangChain chat model instance ready for `.invoke()`.

    Raises:
        ValueError: If no valid API key is found for cloud providers,
                    or if an unsupported provider string is specified.

    Example:
        >>> llm = get_llm(temperature=0.1)
        >>> response = llm.invoke("What is quantum computing?")
        >>> print(response.content)
    """
    # ──────────────────────────────────────────────────────────
    # STEP 1: Determine which provider to use.
    # Priority: Environment variable > Default constant
    # The Streamlit sidebar in app.py sets os.environ["LLM_PROVIDER"]
    # when the user selects a provider from the dropdown.
    # ──────────────────────────────────────────────────────────
    provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).lower()

    # ──────────────────────────────────────────────────────────
    # PROVIDER 1: Groq (Recommended Free Tier)
    #
    # WHY GROQ IS DEFAULT:
    #   - Free tier: 30 requests/min, 14,400 requests/day
    #   - LPU (Language Processing Unit) hardware = ultra-fast inference
    #   - Supports Llama-3.3-70B (large, capable model) at zero cost
    #   - No credit card required to get an API key
    #
    # FALLBACK BEHAVIOR:
    #   If the user hasn't set a GROQ_API_KEY, we silently fall through
    #   to the Google provider as a graceful degradation strategy.
    # ──────────────────────────────────────────────────────────
    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not found. Falling back to Google Gemini provider.")
            provider = "google"  # Graceful fallback — don't crash, try next provider
        else:
            model = model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            logger.info(f"Initializing Groq LLM: model={model}, temperature={temperature}")
            return ChatGroq(
                model=model,
                temperature=temperature,
                groq_api_key=api_key,
                max_retries=3  # Automatic retry on transient API errors (429, 503)
            )

    # ──────────────────────────────────────────────────────────
    # PROVIDER 2: Google Gemini
    #
    # WHY GEMINI AS FALLBACK:
    #   - Free tier available via AI Studio (aistudio.google.com)
    #   - Gemini 1.5 Flash is fast and cost-effective
    #   - Good multilingual support
    #
    # SAFETY CHECK: The "2.0-flash" model name is intercepted and
    # downgraded to "1.5-flash" because the 2.0 variant may not
    # be available in all regions or API versions.
    # ──────────────────────────────────────────────────────────
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "API Key missing! Please enter a GROQ_API_KEY or GOOGLE_API_KEY "
                "in the sidebar or .env file."
            )
        model = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

        # Guard against unavailable model versions
        if "2.0-flash" in model:
            logger.warning(f"Model '{model}' redirected to 'gemini-1.5-flash' for compatibility.")
            model = "gemini-1.5-flash"

        logger.info(f"Initializing Google Gemini LLM: model={model}, temperature={temperature}")
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key,
            max_retries=5  # Higher retries for Gemini (occasional quota spikes)
        )

    # ──────────────────────────────────────────────────────────
    # PROVIDER 3: Ollama (100% Local Offline)
    #
    # WHY OLLAMA EXISTS AS AN OPTION:
    #   - Zero internet dependency — works on airplane mode
    #   - Complete data privacy — nothing leaves your machine
    #   - Great for testing/development when cloud APIs are down
    #
    # PREREQUISITE: User must have `ollama serve` running locally
    # and the model pulled (e.g., `ollama pull llama3.2`)
    # ──────────────────────────────────────────────────────────
    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        model = model_name or os.getenv("OLLAMA_MODEL", "llama3.2")
        logger.info(f"Initializing Ollama LLM: model={model}, temperature={temperature}")
        return ChatOllama(
            model=model,
            temperature=temperature
        )

    # If none of the above providers matched, raise a clear error
    raise ValueError(f"Unsupported LLM provider: '{provider}'. Choose 'groq', 'google', or 'ollama'.")
