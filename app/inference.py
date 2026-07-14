import os
import numpy as np
from PIL import Image
from transformers import CLIPProcessor
from app.config import settings

# Enable verbose logging for Flex delegate debugging
os.environ["TF_CPP_VLOG_LEVEL"] = "1"

# ---------------------------------------------------------------------
# Try to use the new LiteRT interpreter (ai_edge_litert)
# ---------------------------------------------------------------------
try:
    # The Interpreter class is in ai_edge_litert.interpreter
    from ai_edge_litert.interpreter import Interpreter as LiteRTInterpreter

    # The Flex delegate can be imported from ai_edge_litert.delegates
    try:
        from ai_edge_litert.delegates import FlexDelegate
        FLEX_DELEGATE_CLASS = FlexDelegate
    except ImportError:
        # Fallback: try ai_edge_litert.lite.load_delegate
        from ai_edge_litert.lite import load_delegate
        FLEX_DELEGATE_CLASS = None

    USE_LITERT = True
    print("Using ai_edge_litert interpreter.")

except ImportError as e:
    USE_LITERT = False
    import tensorflow as tf
    print(f"Falling back to TensorFlow Lite. Error: {e}")

# ---------------------------------------------------------------------
# Helper to find the Flex delegate library
# ---------------------------------------------------------------------
def _find_flex_delegate():
    """Locate the Flex delegate shared library."""
    if USE_LITERT:
        # For LiteRT, we don't need a separate library; we use the FlexDelegate class.
        return None

    import tensorflow as tf
    lib_path = tf.sysconfig.get_lib()
    candidates = [
        os.path.join(lib_path, 'libtensorflowlite_flex.so'),
        os.path.join(lib_path, 'libflex_delegate.so'),
        os.path.join(lib_path, 'libtensorflowlite_flex.so.2'),
        os.path.join(lib_path, 'python', 'libtensorflowlite_flex.so'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

# ---------------------------------------------------------------------
# Global variables
# ---------------------------------------------------------------------
_interp_image = None
_interp_text = None
_interp_clip = None
_processor = None

def _load_interpreter(model_path: str):
    """Load the interpreter with Flex delegate support."""
    if USE_LITERT:
        # Create interpreter with Flex delegate if available
        try:
            if FLEX_DELEGATE_CLASS is not None:
                # Instantiate the Flex delegate
                flex_delegate = FLEX_DELEGATE_CLASS()
                interpreter = LiteRTInterpreter(model_path=model_path, delegates=[flex_delegate])
                print(f"Loaded LiteRT interpreter with FlexDelegate for {model_path}")
            else:
                # Use load_delegate
                flex_delegate = load_delegate('libtensorflowlite_flex.so')
                interpreter = LiteRTInterpreter(model_path=model_path, delegates=[flex_delegate])
                print(f"Loaded LiteRT interpreter with load_delegate for {model_path}")
        except Exception as e:
            print(f"Warning: Could not load Flex delegate for LiteRT: {e}. Trying without delegate.")
            interpreter = LiteRTInterpreter(model_path=model_path)
        return interpreter
    else:
        # TensorFlow Lite approach
        import tensorflow as tf
        delegate_path = _find_flex_delegate()
        delegate = None
        if delegate_path:
            try:
                delegate = tf.lite.experimental.load_delegate(delegate_path)
                print(f"Loaded Flex delegate from {delegate_path}")
            except Exception as e:
                print(f"Failed to load delegate from {delegate_path}: {e}")
        else:
            print("No Flex delegate found; interpreter may fail for Flex ops.")
        return tf.lite.Interpreter(
            model_path=model_path,
            experimental_delegates=[delegate] if delegate else None
        )

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

def _run_interpreter(interpreter, input_data: list) -> np.ndarray:
    """Set inputs, run inference, return output."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i, detail in enumerate(input_details):
        # Cast to expected dtype
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
    output = _run_interpreter(_interp_clip, [input_ids, attention_mask])
    embedding = output / np.linalg.norm(output, axis=-1, keepdims=True)
    return embedding

def run_style_transfer_image(content_pil: Image.Image, style_pil: Image.Image) -> Image.Image:
    load_models()
    content = _preprocess_image(content_pil)
    style = _preprocess_image(style_pil)
    output = _run_interpreter(_interp_image, [content, style])
    output = np.clip(output[0], 0.0, 1.0) * 255.0
    return Image.fromarray(output.astype(np.uint8))

def run_style_transfer_text(content_pil: Image.Image, style_prompt: str) -> Image.Image:
    load_models()
    content = _preprocess_image(content_pil)
    text_emb = _get_text_embedding(style_prompt)
    output = _run_interpreter(_interp_text, [content, text_emb])
    output = np.clip(output[0], 0.0, 1.0) * 255.0
    return Image.fromarray(output.astype(np.uint8))
