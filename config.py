# config.py
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # AI providers — set ONE
    GROQ_API_KEY:       Optional[str] = None   # FREE — console.groq.com
    GEMINI_API_KEY:     Optional[str] = None   # FREE — aistudio.google.com
    ANTHROPIC_API_KEY:  Optional[str] = None   # $5 credit
    OPENAI_API_KEY:     Optional[str] = None   # needs card

    # RAG settings
    INITIAL_TOP_K:  int   = 20
    FINAL_TOP_K:    int   = 4
    CHUNK_SIZE:     int   = 500
    CHUNK_OVERLAP:  int   = 50
    MAX_TOKENS:     int   = 1024
    TEMPERATURE:    float = 0.1

    APP_TITLE: str = "Company Brain"

    # Telegram (optional)
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    API_BASE_URL:       str = "http://localhost:8000"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def ai_provider(self) -> str:
        if self.GROQ_API_KEY:       return "groq"
        if self.GEMINI_API_KEY:     return "gemini"
        if self.ANTHROPIC_API_KEY:  return "anthropic"
        if self.OPENAI_API_KEY:     return "openai"
        return "none"


settings = Settings()
