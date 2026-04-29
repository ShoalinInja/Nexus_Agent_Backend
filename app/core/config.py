from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    OPENAI_API_KEY: str = ""
    DEV_SECRET_KEY: str = ""
    JWT_SECRET: str = ""
    JWT_EXPIRE_DAYS: int = 30
    ALGORITHM : str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"  # silently ignore unknown env vars (e.g. legacy ANTHROPIC_API_KEY)


settings = Settings()
