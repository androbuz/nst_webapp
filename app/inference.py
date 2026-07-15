import os
import numpy as np
from PIL import Image
from transformers import CLIPProcessor
from app.config import settings
import tensorflow as tf
from tflite_support import flex_delegate

# Global interpreters
_interp_image = None
_interp_text = None
_interp_clip = None
_processor = None

def _load_interpreter(model_path: str):
    """Load TFLite interpreter with Flex delegate."""
    flex_del = flex_delegate.FlexDelegate()
    interpreter = tf.lite.Interpreter(
        model_path=model_path,
        experimental_delegates=[flex_del]
    )
    print(f"Loaded interpreter with FlexDelegate for {model_path}")
    return interpreter

def load_models():
    global _interp_image, _interp_text, _interp_clip
    if _interp_image is None:
        _interp_image = _load_interpreter(settings.IMAGE_GUIDED_MODEL)
        _interp_image.allocate_tensors()
    if _interp_text is None:
        _interp_text = _load_interpreter(settings.TEXT_GUIDED_MODEL)
        _interp_text.allocate_tensors()
    if _interp_clip is None:
        _interp_clip = _load_interpreter(settings.CLIP_ENCODER_MODEL)
        _interp_clip.allocate_tensors()

def _run_tflite(interpreter, input_data: list) -> np.ndarray:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i, detail in enumerate(input_details):
        expected_dtype = detail['dtype']
        input_data[i] = input_data[i].astype(expected_dtype)
        interpreter.set_tensor(detail['index'], input_data[i])

    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

def _preprocess_image(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB").resize((settings.TARGET_SIZE, settings.TARGET_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def _get_text_embedding(prompt: str) -> np.ndarray:
    global _processor
    if _processor is None:
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = _processor(
        text=prompt,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=settings.CLIP_MAX_LENGTH,
    )
    input_ids = inputs["input_ids"].astype(np.int32)
    attention_mask = inputs["attention_mask"].astype(np.int32)
    output = _run_tflite(_interp_clip, [input_ids, attention_mask])
    embedding = output / np.linalg.norm(output, axis=-1, keepdims=True)
    return embedding

def run_style_transfer_image(content_pil: Image.Image, style_pil: Image.Image) -> Image.Image:
    load_models()
    content = _preprocess_image(content_pil)
    style = _preprocess_image(style_pil)
    output = _run_tflite(_interp_image, [content, style])
    output = np.clip(output[0], 0.0, 1.0) * 255.0
    return Image.fromarray(output.astype(np.uint8))

def run_style_transfer_text(content_pil: Image.Image, style_prompt: str) -> Image.Image:
    load_models()
    content = _preprocess_image(content_pil)
    text_emb = _get_text_embedding(style_prompt)
    output = _run_tflite(_interp_text, [content, text_emb])
    output = np.clip(output[0], 0.0, 1.0) * 255.0
    return Image.fromarray(output.astype(np.uint8))
