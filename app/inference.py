import tensorflow as tf
import numpy as np
from PIL import Image
from transformers import CLIPProcessor
from app.config import settings

# Global TFLite interpreters
_interp_image = None
_interp_text = None
_interp_clip = None
_processor = None

def _load_interpreter(model_path: str) -> tf.lite.Interpreter:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def _get_clip_processor():
    global _processor
    if _processor is None:
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _processor

def load_models():
    """Load all TFLite interpreters (called once at startup)."""
    global _interp_image, _interp_text, _interp_clip
    if _interp_image is None:
        _interp_image = _load_interpreter(settings.IMAGE_GUIDED_MODEL)
    if _interp_text is None:
        _interp_text = _load_interpreter(settings.TEXT_GUIDED_MODEL)
    if _interp_clip is None:
        _interp_clip = _load_interpreter(settings.CLIP_ENCODER_MODEL)

def _preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL to float32 array [1, 256, 256, 3] normalized to [0,1]."""
    img = pil_image.convert("RGB")
    img = img.resize((settings.TARGET_SIZE, settings.TARGET_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # (1, 256, 256, 3)

def _run_tflite(interpreter: tf.lite.Interpreter, input_data: list) -> np.ndarray:
    """Run inference and return output tensor."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i, detail in enumerate(input_details):
        interpreter.set_tensor(detail["index"], input_data[i])

    interpreter.invoke()

    # Assume single output
    return interpreter.get_tensor(output_details[0]["index"])

def _get_text_embedding(prompt: str) -> np.ndarray:
    """Tokenize and encode text using quantized CLIP TFLite model."""
    processor = _get_clip_processor()
    inputs = processor(
        text=prompt,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=settings.CLIP_MAX_LENGTH,
    )
    input_ids = inputs["input_ids"].astype(np.int32)          # (1, 77)
    attention_mask = inputs["attention_mask"].astype(np.int32) # (1, 77)

    # Run CLIP encoder
    output = _run_tflite(_interp_clip, [input_ids, attention_mask])
    # Normalize the embedding (L2 norm)
    embedding = output / np.linalg.norm(output, axis=-1, keepdims=True)
    return embedding   # (1, 512)

# ----- Public inference functions -----

def run_style_transfer_image(content_pil: Image.Image, style_pil: Image.Image) -> Image.Image:
    """
    Image‑guided style transfer.
    Returns PIL Image (RGB, 256x256).
    """
    load_models()
    content = _preprocess_image(content_pil)
    style   = _preprocess_image(style_pil)

    # Run the image‑guided TFLite model
    output = _run_tflite(_interp_image, [content, style])   # (1, 256, 256, 3)

    # Clip and convert to uint8
    output = np.clip(output[0], 0.0, 1.0) * 255.0
    output = output.astype(np.uint8)
    return Image.fromarray(output)

def run_style_transfer_text(content_pil: Image.Image, style_prompt: str) -> Image.Image:
    """
    Text‑guided style transfer.
    Returns PIL Image (RGB, 256x256).
    """
    load_models()
    content = _preprocess_image(content_pil)
    text_emb = _get_text_embedding(style_prompt)   # (1, 512)

    # Run the text‑guided TFLite model
    output = _run_tflite(_interp_text, [content, text_emb])   # (1, 256, 256, 3)

    output = np.clip(output[0], 0.0, 1.0) * 255.0
    output = output.astype(np.uint8)
    return Image.fromarray(output)