"""
config.py
---------
Centralized, environment-driven configuration using pydantic-settings.
All secrets/config come from environment variables or a .env file — never hardcoded.
"""

from functools import lru_cache
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- General ---
    APP_NAME: str = "Sentinel ML-IDS Backend"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False

    # --- Security ---
    # JWT signing secret. MUST be overridden in production via env var.
    JWT_SECRET_KEY: str = Field(..., min_length=32)
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Static API key for machine-to-machine sensor ingestion (in addition to JWT for users).
    SENSOR_API_KEY: str = Field(..., min_length=16)

    # Comma-separated list in .env, parsed into a list here.
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    # Rate limiting (requests per minute) per client IP.
    RATE_LIMIT_DEFAULT: str = "60/minute"
    RATE_LIMIT_INGEST: str = "300/minute"

    # --- ML Model ---
    MODEL_PATH: str = "models/sentinel_autoencoder.pt"
    MODEL_DEVICE: Literal["auto", "cpu", "cuda"] = "auto"
    MODEL_INPUT_DIM: int = 32
    MODEL_LATENT_DIM: int = 8
    ANOMALY_THRESHOLD: float = 0.15
    INFERENCE_BATCH_SIZE: int = 64
    INFERENCE_BATCH_TIMEOUT_MS: int = 25  # max wait before flushing a partial batch

    # --- Firebase / Firestore ---
    FIREBASE_ENABLED: bool = False
    FIREBASE_CREDENTIALS_PATH: str = "secrets/firebase-service-account.json"
    FIREBASE_PROJECT_ID: str = ""

    # --- Local fallback persistence ---
    SQLITE_PATH: str = "data/sentinel_cache.db"
    RING_BUFFER_SIZE: int = 5000  # in-memory fallback if disk is unavailable (e.g. read-only IoT fs)
    SYNC_RETRY_SECONDS: int = 15  # how often to retry flushing cached data to Firestore

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def split_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — avoids re-parsing .env on every request."""
    return Settings()
