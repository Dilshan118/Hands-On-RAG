import os
import time
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

# Load environment variables
load_dotenv()

# Configuration Settings
DEFAULT_PROVIDER = "groq"
DEFAULT_MODEL_NAME = "llama-3.3-70b-versatile"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
MAX_REVISIONS = 2
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

def get_llm(model_name: str = None, temperature: float = 0.2) -> BaseChatModel:
    """
    Initialize and return the LLM instance supporting Groq, Google Gemini, and Ollama.
    """
    provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).lower()
    
    # 1. Groq Provider (Recommended Free Tier - Fast, High Quota)
    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            # Fallback to Google if no Groq key set
            provider = "google"
        else:
            model = model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            return ChatGroq(
                model=model,
                temperature=temperature,
                groq_api_key=api_key,
                max_retries=3
            )

    # 2. Google Gemini Provider
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "API Key missing! Please enter a GROQ_API_KEY or GOOGLE_API_KEY in the sidebar or .env file."
            )
        model = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        if "2.0-flash" in model:
            model = "gemini-1.5-flash"
            
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key,
            max_retries=5
        )

    # 3. Local Ollama Provider (100% Offline)
    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        model = model_name or os.getenv("OLLAMA_MODEL", "llama3.2")
        return ChatOllama(
            model=model,
            temperature=temperature
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
