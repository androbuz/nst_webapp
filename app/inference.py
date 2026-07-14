Introduction: 
Comparison of traditional machine learning (pg. 73) vs nst
talk about nst falling under generation rather than classification or regression

mention where techniques came from using references from notebook
	mention NPR in the NST A review


We can mention that currently, I haven't implemented Quantitative Benchmarking nor Qualitative & Saliency Verification.

Okay, I need to write a feature prototype section in the writeup. Write it extensively for me. It should be written in paragraphs (no bullet points or something). Also, I need a MLOps pipeline architecture image; refer to how we are deploying our application here and give me the code to generate the image in python (I will use colab notebook). And leave a placeholder for the MLOps pipeline. I want an extensive report.  

Okay, I need to mention this in the limitations of the project. Write a paragraph or two explaining the limitation and measures taken. 


Mentions:
- use of decorators and custom model definition repetitions in exporting models


Ideas
----
use csvlogger callbacks and put it in the report either as a table
----

Code references:
- all the code for exporting model (find a relevant source)
- total variational loss (ltv) ?




import tensorflow as tf
import numpy as np
from PIL import Image
from transformers import CLIPProcessor
from app.config import settings
import os

# Global interpreters
_interp_image = None
_interp_text = None
_interp_clip = None
_processor = None

def _load_interpreter(model_path: str) -> tf.lite.Interpreter:
    """Load TFLite interpreter with Flex delegate support."""
    # On Linux, the Flex delegate library is named 'libflex_delegate.so'
    # If it fails, we fall back to the default interpreter.
    try:
        flex_delegate = tf.lite.experimental.load_delegate(
            tf.lite.experimental.load_delegate(
                os.path.join(tf.sysconfig.get_lib(), 'libflex_delegate.so')
            )
        )
    except Exception as e:
        # Fallback: try without delegate (may still work for some ops)
        print(f"Warning: Could not load Flex delegate: {e}. Falling back to default interpreter.")
        interpreter = tf.lite.Interpreter(model_path=model_path)
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

def _get_clip_processor():
    global _processor
    if _processor is None:
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _processor

def _preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Resize to 256x256 and normalize to [0,1]."""
    img = pil_image.convert("RGB").resize((settings.TARGET_SIZE, settings.TARGET_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # (1, 256, 256, 3)

def _run_tflite(interpreter: tf.lite.Interpreter, input_data: list) -> np.ndarray:
    """Set inputs, run inference, return output."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i, detail in enumerate(input_details):
        interpreter.set_tensor(detail["index"], input_data[i])

    interpreter.invoke()
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
    input_ids = inputs["input_ids"].astype(np.int32)
    attention_mask = inputs["attention_mask"].astype(np.int32)
    output = _run_tflite(_interp_clip, [input_ids, attention_mask])
    # Normalize
    embedding = output / np.linalg.norm(output, axis=-1, keepdims=True)
    return embedding   # shape (1, 512)

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
