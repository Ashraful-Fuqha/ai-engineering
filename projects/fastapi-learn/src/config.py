from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_base_url: str = "https://jsonplaceholder.typicode.com"
    api_timeout_seconds: float = 5.0
    database_url: str = "posts.db"

    model_config = SettingsConfigDict(
        env_file=("../../.env", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
