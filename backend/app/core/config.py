"""
Application configuration module.
Loads settings from environment variables and .env file using Pydantic Settings.
"""

from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Core application settings with strict validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # General App Info
    APP_NAME: str = "AgentX"
    APP_VERSION: str = "0.1.0"
    APP_DESCRIPTION: str = "Autonomous Multi-Agent Research & Intelligence System"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"

    # Server Binding
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Security & CORS
    SECRET_KEY: str = "agentx-default-secret-change-in-production"
    ALLOWED_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        import json
        try:
            return json.loads(v)
        except Exception:
            return [v]

    # LLM Configuration (OpenAI)
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API Key")
    OPENAI_MODEL: str = Field(
        default="gpt-4o", description="Default model name, configurable via ENV"
    )
    OPENAI_TEMPERATURE: float = 0.2

    # Search Configuration (Tavily)
    TAVILY_API_KEY: str = Field(default="", description="Tavily Search API Key")

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/agentx"

    # Vector Database
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_data"

    # Redis Cache & Message Queue
    REDIS_URL: str = "redis://localhost:6379/0"

    # Workflow Limits & Guardrails
    MAX_SEARCH_RESULTS_PER_TASK: int = 5
    MAX_REVISION_CYCLES: int = 3
    MAX_CONCURRENT_RESEARCHERS: int = 4
    WORKFLOW_TIMEOUT_SECONDS: int = 600


# Global cached settings instance
settings = Settings()
