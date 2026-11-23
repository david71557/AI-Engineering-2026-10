from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Session 2 fields (kept for backwards compatibility with the live demos) ---
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    LLM_PROVIDER: Literal["openai", "anthropic"] = "anthropic"
    LLM_MODEL: str = "claude-haiku-4-5"
    APP_ENV: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "DEBUG"

    # --- Session 3 fields (LiteLLM wrapper, Redis cache, Streamlit transport) ---
    PRIMARY_MODEL: str = "gpt-4o-mini"
    FALLBACK_MODEL: str = "claude-haiku-4-5-20251001"
    LLM_TIMEOUT: int = 30
    LLM_RETRIES: int = 2

    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 86400

    # --- Session 4 fields (semantic cache, needs Redis Stack) ---
    SEMANTIC_CACHE_ENABLED: bool = True
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    SEMANTIC_CACHE_THRESHOLD: float = 0.85
    SEMANTIC_CACHE_TTL: int = 86400
    # When True, the semantic cache LOGS potential hits but does NOT serve them.
    # Used to gather metrics before flipping the cache on in production.
    SEMANTIC_CACHE_LOG_ONLY: bool = False

    ESTIMATOR_API_BASE_URL: str = "http://localhost:8000"

    # --- Session 5 fields (conversational memory + attachments) ---
    # MAX_CONVERSATION_TURNS counts user+assistant pairs. The system prompt is
    # always preserved as an invariant on top of the window.
    MAX_CONVERSATION_TURNS: int = 6
    # Hard cap per extracted attachment (in characters) to protect the context
    # window. Real chunking enters in module 3.
    MAX_ATTACHMENT_CHARS: int = 60_000
    # The metadata extractor runs once per turn; a small/cheap model is enough.
    METADATA_EXTRACTOR_MODEL: str = "gpt-4o-mini"

    # --- Session 5 live: compression + dynamic tier + Actor-Critic-Boss ---
    # Anchor detector: "heuristic" (regex over key phrases, no LLM call) or
    # "llm" (binary classifier on COMPRESSION_MODEL). Heuristic is the default for cost.
    ANCHOR_DETECTION_MODE: Literal["heuristic", "llm"] = "heuristic"
    # Cheap model used by the cumulative summarizer (history compression).
    COMPRESSION_MODEL: str = "gpt-4o-mini"
    # Conversational prompt version (app/prompts/conversation/<version>/).
    # v1 = metadata + current estimate. v2 = adds the <audience> block driven
    # by the resolved tier and the <critic_feedback> block used by the Boss.
    CONVERSATIONAL_PROMPT_VERSION: str = "v2"
    # Critic model (read-only auditor; cheap is fine).
    CRITIC_MODEL: str = "gpt-4o-mini"
    # Max Boss iterations (each = 1 actor + 1 critic call). Three is the
    # practical floor: one draft + two directed retries; with two, the actor
    # often can't address every flagged issue in its single retry.
    BOSS_MAX_ITERATIONS: int = 3

    # --- Session 6 fields (data-driven AI: persistence + ingestion + PII) ---
    # Postgres connection string. pgvector/pgvector:pg16 image; the extension
    # is dormant in S06 (no CREATE EXTENSION vector) and only activates in S07.
    DATABASE_URL: str = (
        "postgresql+psycopg://estimator:estimator@localhost:5433/estimator"
    )
    # Where the YAML catalog lives. Resolved relative to the working directory.
    CATALOG_PATH: Path = Path("data/catalog/catalog.yaml")
    # Root where ``CatalogSource.location`` entries are resolved against.
    INGESTION_DATA_ROOT: Path = Path("data/seed")
    # Output folder for the cleaned, normalized budget JSON files.
    PROCESSED_DATA_DIR: Path = Path("data/processed")
    # spaCy model loaded by the Presidio AnalyzerEngine (Spanish).
    PRESIDIO_SPACY_MODEL: str = "es_core_news_md"
    # Locale used by Faker to generate consistent pseudonyms per entity_type.
    PSEUDONYM_FAKER_LOCALE: str = "es_ES"
    # HMAC salt. Stored in env so it can be rotated independently of the code.
    PSEUDONYM_HASH_SALT: str = "change-me-in-prod"

    @model_validator(mode="after")
    def validate_at_least_one_api_key(self) -> "Settings":
        """LiteLLM may try either provider via fallback, so we require at least one key."""
        if not self.OPENAI_API_KEY and not self.ANTHROPIC_API_KEY:
            raise ValueError(
                "At least one of OPENAI_API_KEY or ANTHROPIC_API_KEY must be set"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings (singleton)."""
    return Settings()
