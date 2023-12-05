from enum import Enum
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class GPTModel(str, Enum):
    gpt_4 = "gpt-4"
    gpt_3_5_turbo = "gpt-3.5-turbo"

class Settings(BaseSettings):
    service_name: str = "Análisis del sentimiento API"
    k_revision: str = "Local"
    log_level: str = "DEBUG"
    openai_key: str
    model: GPTModel = GPTModel.gpt_4

    class Config:
        env_file = ".env"

@lru_cache
def get_settings():
    return Settings()