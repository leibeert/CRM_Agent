from functools import lru_cache
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    def __init__(self):
        # API Settings
        self.API_V1_STR = "/api/v1"
        self.PROJECT_NAME = "AI Recruitment System"
        
        # Database Settings
        self.DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://user:password@localhost/recruitment_crm")
        
        # OpenAI Settings
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo-preview")
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
        
        # Security Settings
        self.SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
        self.ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 8  # 8 days
        
        # CORS Settings
        self.BACKEND_CORS_ORIGINS = ["*"]

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings() 