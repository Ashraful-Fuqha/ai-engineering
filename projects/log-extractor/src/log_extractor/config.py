from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from environment variables or .env file."""
    
    gemini_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"),
        description="API key for authentication with Google AI Studio",
    )
    
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai",
        validation_alias=AliasChoices("GEMINI_BASE_URL", "OPENAI_BASE_URL"),
        description="Base URL for OpenAI-compatible endpoint",
    )
    
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        validation_alias=AliasChoices("GEMINI_MODEL", "MODEL"),
        description="Target model name",
    )
    
    default_timeout_seconds: float = Field(
        default=45.0,
        validation_alias=AliasChoices("DEFAULT_TIMEOUT_SECONDS", "TIMEOUT"),
        description="Network timeout in seconds",
    )

    model_config = SettingsConfigDict(
        env_file=("../../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

