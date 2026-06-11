from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-5", alias="OPENAI_MODEL")
    openai_wire_api: str = Field(default="chat_completions", alias="OPENAI_WIRE_API")
    openai_timeout: float = Field(default=120.0, alias="OPENAI_TIMEOUT")
    metabase_base_url: str = Field(default="", alias="METABASE_BASE_URL")
    metabase_api_key: str = Field(default="", alias="METABASE_API_KEY")
    agent_dry_run: bool = Field(default=True, alias="AGENT_DRY_RUN")
    agent_mode: str = Field(default="pipeline", alias="AGENT_MODE")
    agent_require_token: bool = Field(default=False, alias="AGENT_REQUIRE_TOKEN")
    agent_memory_path: str = Field(default=".metabase_agent_memory.json", alias="AGENT_MEMORY_PATH")
    agent_state_path: str = Field(default=".metabase_agent_state.json", alias="AGENT_STATE_PATH")
    agent_api_token: str = Field(default="", alias="AGENT_API_TOKEN")
    metabase_bigquery_database_id: int = Field(default=19, alias="METABASE_BIGQUERY_DATABASE_ID")
    agent_report_range_start: str = Field(default="2025-11-01", alias="AGENT_REPORT_RANGE_START")
    agent_report_range_end_exclusive: str = Field(default="2026-05-01", alias="AGENT_REPORT_RANGE_END_EXCLUSIVE")
    agent_report_timezone: str = Field(default="US/Pacific", alias="AGENT_REPORT_TIMEZONE")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
