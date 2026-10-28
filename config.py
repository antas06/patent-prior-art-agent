import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
  # API Keys
  CODIV_API_KEY: str = os.getenv("CODIV_API_KEY", "")
  GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
  
  # Langfuse Observability
  LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
  LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
  # Accepts LANGFUSE_HOST (preferred) with LANGFUSE_BASE_URL as a fallback
  LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"
  
  # Models
  SYSTEM1_MODEL: str = "laya-1.0"
  LLM_MODEL: str = "openai/gpt-oss-120b"
  
  # System 1 CRAG Thresholds
  THRESH_RELEVANT: float = 0.65
  THRESH_IRRELEVANT: float = 0.70
  MIN_RELEVANT_CHUNKS: int = 2
  
  # Engine Limits
  MAX_FEATURES: int = 4
  MAX_WEB_ROUNDS: int = 2
  WEB_FALLBACK_THRESHOLD: float = 0.4
  QDRANT_PATH: str = "./data/qdrant_db"