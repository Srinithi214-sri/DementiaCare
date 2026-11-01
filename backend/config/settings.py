import os
import sys
import logging
from dotenv import load_dotenv, find_dotenv

class Settings:
    def __init__(self):
        # Load from .env if present
        load_dotenv(find_dotenv())
        
        self.MONGODB_URI = os.getenv("MONGO_URI") or os.getenv("MONGODB_URI")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.GEMINI_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT", "10.0"))
        
        self.JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
        self.JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
        self.JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        
        self.CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
        
        self.EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.SEMANTIC_TOP_K = int(os.getenv("SEMANTIC_TOP_K", "5"))
        self.SEMANTIC_MIN_SCORE = float(os.getenv("SEMANTIC_MIN_SCORE", "0.3"))
        
        self.validate()

    def validate(self):
        if not self.MONGODB_URI:
            logging.error("MONGODB_URI environment variable is required but not set.")
            sys.exit("Database configuration error: Missing connection string.")
            
        if not self.JWT_SECRET_KEY:
            logging.error("JWT_SECRET_KEY is missing. Authentication is unavailable and insecure.")
            sys.exit("Security configuration error: JWT_SECRET_KEY must be provided.")
            
        if self.JWT_SECRET_KEY in ["secret", "test", "password", "123456"]:
            logging.error("JWT_SECRET_KEY is set to a wildly insecure predictable value.")
            sys.exit("Security configuration error: JWT_SECRET_KEY is insecure.")
            
        if not self.GEMINI_API_KEY:
            logging.warning("GEMINI_API_KEY environment variable is not set. Gemini integration is disabled, deterministic fallback will be used.")

    def get_cors_origins(self):
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

# Create a global settings instance
settings = Settings()
