"""
LLM Factory Service for AgentX.
Provides configurable ChatOpenAI instances without hardcoding model names.
"""

from typing import Optional
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("core.llm")


class LLMConfigurationError(Exception):
    """Raised when LLM configuration or credentials are invalid."""
    pass


def get_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    streaming: bool = False,
    timeout: Optional[float] = None,
    max_retries: int = 3,
) -> ChatOpenAI:
    """
    Construct and return a configured ChatOpenAI client.
    
    Model selection priority:
    1. Explicit model parameter passed to get_llm
    2. settings.OPENAI_MODEL environment variable
    3. Default fallback "gpt-4o"
    """
    api_key = settings.OPENAI_API_KEY
    model_name = model or settings.OPENAI_MODEL or "gpt-4o"
    temp = temperature if temperature is not None else settings.OPENAI_TEMPERATURE

    if not api_key:
        logger.warning(
            "OPENAI_API_KEY is not set. Real model calls will fail until configured."
        )

    logger.debug(f"Initializing ChatOpenAI with model={model_name}, temp={temp}")

    return ChatOpenAI(
        model=model_name,
        temperature=temp,
        api_key=api_key or "sk-placeholder",
        streaming=streaming,
        timeout=timeout or float(settings.WORKFLOW_TIMEOUT_SECONDS),
        max_retries=max_retries,
    )

