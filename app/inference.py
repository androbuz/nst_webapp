import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from transformers import CLIPProcessor

# Force parent directory into path so custom_classes is found
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from app.config import settings
from custom_classes.models import StyleTransferModel, RefinementDecoder
from custom_classes.layers import PatchEmbedding, TransformerEncoder, ContentAwarePositionalEncoding

# Global model and processor instances
_model = None
_clip_text_encoder = None
_processor = None

def load_fp16_weights(model, weights_path):
    """Helper to load weights from .npz FP16 files."""
    data = np.load(weights_path)
    weights = [data[f'arr_{i}'].astype(np.float32) for i in range(len(data.files))]
    model.set_weights(weights)
    print(f"Loaded weights from {weights_path}")

def load_models():
    global _model, _clip_text_encoder, _processor

    if _processor is None:
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    if _model is None:
        projection_dim = 256
        target_size = 256

        c_embedder = PatchEmbedding(patch_size=8, projection_dim=projection_dim)
        s_embedder = PatchEmbedding(patch_size=8, projection_dim=projection_dim)
        c_encoder = tf.keras.Sequential([TransformerEncoder(projection_dim, 8, projection_dim*4) for _ in range(2)])
        s_encoder = tf.keras.Sequential([TransformerEncoder(projection_dim, 8, projection_dim*4) for _ in range(2)])
        cape = ContentAwarePositionalEncoding(target_spatial_size=18, projection_dim=projection_dim)
        decoder = RefinementDecoder(projection_dim=projection_dim, output_image_size=target_size)

        _model = StyleTransferModel(
            content_patch_embedder=c_embedder,
            style_patch_embedder=s_embedder,
            content_encoder=c_encoder,
            style_encoder=s_encoder,
            cape_layer=cape,
            refinement_decoder=decoder,
            projection_dim=projection_dim,
            style_text_embedding_dim=512,
            num_patches=1024
        )

        if os.path.exists(settings.MODEL_WEIGHTS_PATH):
            if settings.MODEL_WEIGHTS_PATH.endswith('.npz'):
                load_fp16_weights(_model, settings.MODEL_WEIGHTS_PATH)
            else:
                _model.load_weights(settings.MODEL_WEIGHTS_PATH)

    if _clip_text_encoder is None:
        from keras_cv.models import CLIP
        clip = CLIP.from_preset("clip-vit-base-patch32")
        _clip_text_encoder = clip.text_encoder
        if os.path.exists(settings.CLIP_WEIGHTS_PATH):
             load_fp16_weights(_clip_text_encoder, settings.CLIP_WEIGHTS_PATH)

def _preprocess_image(pil_image):
    img = pil_image.convert("RGB").resize((256, 256))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def _get_text_embedding(prompt):
    inputs = _processor(text=prompt, return_tensors="tf", padding="max_length", truncation=True, max_length=77)
    text_features = _clip_text_encoder(inputs.input_ids, attention_mask=inputs.attention_mask)
    embedding = tf.linalg.normalize(text_features)[0]
    return embedding

def run_style_transfer_image(content_pil, style_pil):
    load_models()
    content = _preprocess_image(content_pil)
    style = _preprocess_image(style_pil)
    output = _model(content_img=content, style_img=style, training=False)
    output = np.clip(output[0].numpy(), 0.0, 1.0) * 255.0
    return Image.fromarray(output.astype(np.uint8))

def run_style_transfer_text(content_pil, style_prompt):
    load_models()
    content = _preprocess_image(content_pil)
    text_emb = _get_text_embedding(style_prompt)
    output = _model(content_img=content, style_text_embedding=text_emb, training=False)
    output = np.clip(output[0].numpy(), 0.0, 1.0) * 255.0
    return Image.fromarray(output.astype(np.uint8))
