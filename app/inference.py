import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from transformers import CLIPProcessor

# Force parent directory into path so custom_classes is found
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from app.config import settings
from custom_classes.models import RefinementDecoder
from custom_classes.layers import PatchEmbedding, TransformerEncoder, ContentAwarePositionalEncoding

# Global model and processor instances
_model_image = None
_model_video = None
_clip_text_encoder = None
_processor = None

def load_fp16_weights(model, weights_path):
    """Helper to load weights from .npz FP16 files."""
    data = np.load(weights_path)
    weights = [data[f'arr_{i}'].astype(np.float32) for i in range(len(data.files))]
    model.set_weights(weights)
    print(f"Loaded weights from {weights_path}")

def build_hybrid_style_transfer_model(content_shape=(256, 256, 3), style_shape=(256, 256, 3), is_video=False):
    import keras
    
    projection_dim = 256
    patch_size = 8
    CLIP_EMBEDDING_DIM = 512
    num_encoder_blocks = 2
    embed_dim = projection_dim
    num_heads_content = 16
    num_heads_style = 4
    ffn_units = embed_dim * 4

    if is_video:
        c_img_in = keras.Input(shape=(2,) + content_shape, name="content_img_in")
        s_img_in = keras.Input(shape=style_shape, name="style_img_in")
    else:
        c_img_in = keras.Input(shape=content_shape, name="content_img_in")
        s_img_in = keras.Input(shape=style_shape, name="style_img_in")

    t_emb_in = keras.Input(shape=(CLIP_EMBEDDING_DIM,), name="style_text_embedding_in")

    content_patcher = PatchEmbedding(patch_size, projection_dim, name="content_patcher")
    style_patcher = PatchEmbedding(patch_size, projection_dim, name="style_patcher")
    cape_layer = ContentAwarePositionalEncoding(projection_dim=projection_dim, name="cape_layer")
    decoder = RefinementDecoder(projection_dim=projection_dim, output_image_size=content_shape[0], name="decoder")
    text_style_projection_layer = keras.layers.Dense(projection_dim, name="text_style_projection")

    # Inner encoders
    content_inputs = keras.Input(shape=(None, projection_dim), name="content_input")
    x_content = content_inputs
    for i in range(num_encoder_blocks):
        x_content = TransformerEncoder(embed_dim, num_heads_content, ffn_units, name=f"content_encoder_block_{i}")(x_content)
    content_encoder = keras.Model(inputs=content_inputs, outputs=x_content, name="content_encoder")

    style_inputs = keras.Input(shape=(None, projection_dim), name="style_input")
    x_style = style_inputs
    for i in range(num_encoder_blocks):
        x_style = TransformerEncoder(embed_dim, num_heads_style, ffn_units, name=f"style_encoder_block_{i}")(x_style)
    style_encoder = keras.Model(inputs=style_inputs, outputs=x_style, name="style_encoder")

    def process_frame(c_img, s_img_input, t_emb_input):
        c_feats = content_encoder(content_patcher(c_img))
        is_text_guided = keras.ops.equal(keras.ops.sum(s_img_input, axis=[1, 2, 3]), 0.0)
        s_feats_from_image = style_encoder(style_patcher(s_img_input))
        spatial_tokens = keras.ops.shape(c_feats)[1]
        t_emb_projected = text_style_projection_layer(t_emb_input)
        t_feats_repeated = keras.layers.Reshape((1, projection_dim))(t_emb_projected)
        t_feats_from_text = keras.layers.Lambda(lambda x: keras.ops.tile(x, [1, spatial_tokens, 1]))(t_feats_repeated)

        condition_expanded = keras.ops.expand_dims(keras.ops.expand_dims(is_text_guided, axis=1), axis=1)
        s_feats_final = keras.ops.where(condition_expanded, t_feats_from_text, s_feats_from_image)

        spatial_h = content_shape[0] // patch_size
        spatial_w = content_shape[1] // patch_size

        c_feats_spatial = keras.layers.Reshape((spatial_h, spatial_w, projection_dim))(c_feats)
        s_feats_spatial = keras.layers.Reshape((spatial_h, spatial_w, projection_dim))(s_feats_final)

        combined = cape_layer([c_feats_spatial, s_feats_spatial])
        output_img = decoder(combined)
        return output_img

    if is_video:
        frame_t = c_img_in[:, 0, ...]
        frame_tm1 = c_img_in[:, 1, ...]
        out_t = process_frame(frame_t, s_img_in, t_emb_in)
        out_tm1 = process_frame(frame_tm1, s_img_in, t_emb_in)
        output_img = keras.ops.stack([out_t, out_tm1], axis=1)
    else:
        output_img = process_frame(c_img_in, s_img_in, t_emb_in)

    return keras.Model(inputs=[c_img_in, s_img_in, t_emb_in], outputs=output_img)

def load_models(is_video=False):
    global _model_image, _model_video, _clip_text_encoder, _processor

    if _processor is None:
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    if _clip_text_encoder is None:
        from keras_cv.models import CLIP
        clip = CLIP.from_preset("clip-vit-base-patch32")
        _clip_text_encoder = clip.text_encoder
        # Dummy call to initialize shape
        dummy_ids = tf.zeros((1, 77), dtype=tf.int32)
        dummy_mask = tf.zeros((1, 77), dtype=tf.int32)
        _ = _clip_text_encoder(dummy_ids, attention_mask=dummy_mask)
        if os.path.exists(settings.CLIP_WEIGHTS_PATH):
             load_fp16_weights(_clip_text_encoder, settings.CLIP_WEIGHTS_PATH)

    if is_video:
        if _model_video is None:
            _model_video = build_hybrid_style_transfer_model(is_video=True)
            if os.path.exists(settings.MODEL_WEIGHTS_PATH):
                load_fp16_weights(_model_video, settings.MODEL_WEIGHTS_PATH)
        return _model_video
    else:
        if _model_image is None:
            _model_image = build_hybrid_style_transfer_model(is_video=False)
            if os.path.exists(settings.MODEL_WEIGHTS_PATH):
                load_fp16_weights(_model_image, settings.MODEL_WEIGHTS_PATH)
        return _model_image

def _preprocess_image(pil_image):
    img = pil_image.convert("RGB").resize((256, 256))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def _get_text_embedding(prompt):
    inputs = _processor(text=prompt, return_tensors="np", padding="max_length", truncation=True, max_length=77)
    input_ids = tf.convert_to_tensor(inputs.input_ids)
    attention_mask = tf.convert_to_tensor(inputs.attention_mask)
    text_features = _clip_text_encoder(input_ids, attention_mask=attention_mask)
    embedding = tf.linalg.normalize(text_features)[0]
    return embedding

def run_style_transfer_image(content_pil, style_pil):
    model = load_models(is_video=False)
    content = _preprocess_image(content_pil)
    style = _preprocess_image(style_pil)
    dummy_text_inf = tf.zeros([tf.shape(content)[0], 512])
    output = model([content, style, dummy_text_inf], training=False)
    output = np.clip(output[0].numpy(), 0.0, 1.0) * 255.0
    return Image.fromarray(output.astype(np.uint8))

def run_style_transfer_text(content_pil, style_prompt):
    model = load_models(is_video=False)
    content = _preprocess_image(content_pil)
    text_emb = _get_text_embedding(style_prompt)
    dummy_style_inf = tf.zeros_like(content)
    output = model([content, dummy_style_inf, text_emb], training=False)
    output = np.clip(output[0].numpy(), 0.0, 1.0) * 255.0
    return Image.fromarray(output.astype(np.uint8))

def run_style_transfer_video(input_video_path, output_video_path, style_pil=None, style_prompt=None):
    model = load_models(is_video=True)
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError("Cannot open input video")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if style_prompt:
        text_emb = _get_text_embedding(style_prompt)
        dummy_style = tf.zeros([1, 256, 256, 3])
    else:
        text_emb = tf.zeros([1, 512])
        dummy_style = _preprocess_image(style_pil)
        
    prev_frame_tensor = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        curr_frame_tensor = _preprocess_image(pil_img)
        
        if prev_frame_tensor is None:
            prev_frame_tensor = curr_frame_tensor
            
        # Stack current and previous frames
        video_input = tf.stack([curr_frame_tensor, prev_frame_tensor], axis=1)
        
        output = model([video_input, dummy_style, text_emb], training=False)
        
        # Retrieve frame_t (index 0)
        styled_frame_tensor = output[0, 0, ...]
        styled_frame_np = np.clip(styled_frame_tensor.numpy(), 0.0, 1.0) * 255.0
        styled_frame_np = cv2.resize(styled_frame_np, (width, height)).astype(np.uint8)
        
        frame_bgr = cv2.cvtColor(styled_frame_np, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        
        prev_frame_tensor = curr_frame_tensor
        
    cap.release()
    out.release()
    return output_video_path
