import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Updated paths for FP16 weights
    MODEL_WEIGHTS_PATH: str = os.getenv("MODEL_WEIGHTS_PATH", "models/hybrid_style_weights_fp16.npz")
    CLIP_WEIGHTS_PATH: str = os.getenv("CLIP_WEIGHTS_PATH", "models/clip_text_weights_fp16.npz")
    
    # Web app settings
    DEBUG: bool = os.getenv("DEBUG", "False") == "True"

settings = Settings()
