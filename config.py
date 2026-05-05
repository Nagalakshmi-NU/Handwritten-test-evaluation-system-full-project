import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key")
    DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:123456@localhost:5432/testpaper_db")
    # Railway PostgreSQL uses postgres:// but SQLAlchemy needs postgresql://
    SQLALCHEMY_DATABASE_URI = DATABASE_URL.replace("postgres://", "postgresql://", 1) if DATABASE_URL else "postgresql://postgres:123456@localhost:5432/testpaper_db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CELERY_BROKER_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = os.environ.get("RAILWAY_ENVIRONMENT") is not None