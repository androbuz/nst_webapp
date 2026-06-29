import os

class Settings:
    # TFLite model paths
    MODEL_DIR = os.getenv("MODEL_DIR", "./models")
    IMAGE_GUIDED_MODEL = os.path.join(MODEL_DIR, "hybrid_style_transfer_model_image_guided_quantized.tflite")
    TEXT_GUIDED_MODEL  = os.path.join(MODEL_DIR, "hybrid_style_transfer_model_text_guided_quantized.tflite")
    CLIP_ENCODER_MODEL = os.path.join(MODEL_DIR, "clip_text_encoder_quantized.tflite")

    # Image pre‑processing
    TARGET_SIZE = 256
    CLIP_MAX_LENGTH = 77

settings = Settings()