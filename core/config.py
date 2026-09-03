"""
Core Configuration Module for WatermarkRemoverAI.
Utilizes Pydantic Settings for type-safe environment variable management.
"""

import os
from pathlib import Path

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    USE_PYDANTIC_V2_SETTINGS = True
except ImportError:
    try:
        from pydantic import BaseSettings
        SettingsConfigDict = None
        USE_PYDANTIC_V2_SETTINGS = False
    except ImportError:
        # Minimal standalone class for fallback
        class BaseSettings:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        SettingsConfigDict = None
        USE_PYDANTIC_V2_SETTINGS = False


class Settings(BaseSettings):
    # Application metadata
    PROJECT_NAME: str = "WatermarkRemoverAI"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False

    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False

    # Directory Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    STORAGE_DIR: Path = BASE_DIR / "storage"

    # Inference & Hardware Configuration
    DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5

    if USE_PYDANTIC_V2_SETTINGS and SettingsConfigDict is not None:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=True,
            extra="ignore"
        )
    else:
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = True
            extra = "ignore"


settings = Settings()
