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
    siliconflow_api_key: str = Field(default="", alias="SILICONFLOW_API_KEY")
    siliconflow_base_url: str = Field(default="https://api.siliconflow.cn/v1", alias="SILICONFLOW_BASE_URL")
    metabase_base_url: str = Field(default="", alias="METABASE_BASE_URL")
    metabase_api_key: str = Field(default="", alias="METABASE_API_KEY")
    agent_dry_run: bool = Field(default=True, alias="AGENT_DRY_RUN")
    agent_mode: str = Field(default="pipeline", alias="AGENT_MODE")
    agent_require_token: bool = Field(default=False, alias="AGENT_REQUIRE_TOKEN")
    agent_memory_path: str = Field(default=".metabase_agent_memory.json", alias="AGENT_MEMORY_PATH")
    agent_state_path: str = Field(default=".metabase_agent_state.json", alias="AGENT_STATE_PATH")
    agent_store: str = Field(default="memory", alias="AGENT_STORE")
    agent_store_mongodb_uri: str = Field(default="", alias="AGENT_STORE_MONGODB_URI")
    agent_store_mongodb_database: str = Field(default="metabase_agent_sessions", alias="AGENT_STORE_MONGODB_DATABASE")
    agent_session_ttl_seconds: float = Field(default=0.0, alias="AGENT_SESSION_TTL_SECONDS")
    agent_api_token: str = Field(default="", alias="AGENT_API_TOKEN")
    agent_checkpoint_backend: str = Field(default="none", alias="AGENT_CHECKPOINT_BACKEND")
    agent_checkpoint_mongodb_uri: str = Field(default="", alias="AGENT_CHECKPOINT_MONGODB_URI")
    agent_checkpoint_mongodb_database: str = Field(default="metabase_agent_checkpoints", alias="AGENT_CHECKPOINT_MONGODB_DATABASE")
    agent_checkpoint_ttl_seconds: int = Field(default=0, alias="AGENT_CHECKPOINT_TTL_SECONDS")
    agent_tenant_id: str = Field(default="default", alias="AGENT_TENANT_ID")
    agent_user_id: str = Field(default="", alias="AGENT_USER_ID")
    agent_long_term_memory_enabled: bool = Field(default=False, alias="AGENT_LONG_TERM_MEMORY_ENABLED")
    agent_memory_llm_extractor: bool = Field(default=False, alias="AGENT_MEMORY_LLM_EXTRACTOR")
    agent_mongodb_uri: str = Field(default="", alias="AGENT_MONGODB_URI")
    agent_mongodb_database: str = Field(default="metabase_agent", alias="AGENT_MONGODB_DATABASE")
    agent_memory_collection: str = Field(default="agent_memories", alias="AGENT_MEMORY_COLLECTION")
    agent_pgvector_dsn: str = Field(default="", alias="AGENT_PGVECTOR_DSN")
    agent_pgvector_table: str = Field(default="memory_embeddings", alias="AGENT_PGVECTOR_TABLE")
    agent_pgvector_auto_create: bool = Field(default=True, alias="AGENT_PGVECTOR_AUTO_CREATE")
    agent_embedding_provider: str = Field(default="hash", alias="AGENT_EMBEDDING_PROVIDER")
    agent_embedding_model: str = Field(default="text-embedding-3-small", alias="AGENT_EMBEDDING_MODEL")
    agent_embedding_dimensions: int = Field(default=1536, alias="AGENT_EMBEDDING_DIMENSIONS")
    agent_skills_enabled: bool = Field(default=True, alias="AGENT_SKILLS_ENABLED")
    agent_skills_path: str = Field(default="skills", alias="AGENT_SKILLS_PATH")
    agent_skills_max_chars: int = Field(default=6000, alias="AGENT_SKILLS_MAX_CHARS")
    metabase_bigquery_database_id: int = Field(default=19, alias="METABASE_BIGQUERY_DATABASE_ID")
    agent_report_range_start: str = Field(default="2025-11-01", alias="AGENT_REPORT_RANGE_START")
    agent_report_range_end_exclusive: str = Field(default="2026-05-01", alias="AGENT_REPORT_RANGE_END_EXCLUSIVE")
    agent_report_timezone: str = Field(default="US/Pacific", alias="AGENT_REPORT_TIMEZONE")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
