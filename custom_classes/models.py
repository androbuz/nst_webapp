import tensorflow as tf
import keras
from keras import layers as klayers

from custom_classes.layers import DecoderBlock

@keras.utils.register_keras_serializable()
class RefinementDecoder(keras.Model):
    # define the decoder model used to reconstruct an output image
    def __init__(self, projection_dim, output_image_size, **kwargs):
        super().__init__(**kwargs)
        # storing the feature dimension and target output image size
        self.projection_dim = projection_dim
        self.output_image_size = output_image_size
        # the number of filters is progressively reduced to refine the image
        filter_configs = [projection_dim, projection_dim // 2, projection_dim // 4]
        # decoder should not use fewer than 32 filters per block
        filter_configs = [max(f, 32) for f in filter_configs]
        # creating the decoder blocks to transform the feature map
        self.decoder_blocks = []
        self.decoder_blocks.append(DecoderBlock(filter_configs[0], name="decoder_block_0"))
        self.decoder_blocks.append(DecoderBlock(filter_configs[1], name="decoder_block_1"))
        self.decoder_blocks.append(DecoderBlock(filter_configs[2], name="decoder_block_2"))
        # converts the final features into an rgb image
        self.final_conv = klayers.Conv2D(3, kernel_size=3, padding='same', activation='sigmoid', name='output_image_conv')

    def call(self, inputs):
        # extract batch size and sequence length from the input
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.shape(inputs)[1]
        # calculates the spatial dimension needed for conversion of feature vectors
        spatial_side = tf.cast(tf.sqrt(tf.cast(sequence_length, tf.float32)), tf.int32)
        # verify that the sequence length can form a square spatial grid
        tf.Assert(tf.equal(spatial_side * spatial_side, sequence_length),
                  ["Input sequence_length must be a perfect square for reshaping to 2D."])
        x = tf.reshape(inputs, (batch_size, spatial_side, spatial_side, self.projection_dim))
        # pass the feature map through each decoder block
        for block in self.decoder_blocks:
            x = block(x)
        output_image = self.final_conv(x)
        return output_image

    def get_config(self):
        config = super().get_config()
        config.update({
            "projection_dim": self.projection_dim,
            "output_image_size": self.output_image_size,
        })
        return config


@keras.utils.register_keras_serializable()
class StyleTransferModel(keras.Model):
    def __init__(self,
                 content_patch_embedder,
                 style_patch_embedder,
                 content_encoder,
                 style_encoder,
                 cape_layer,
                 refinement_decoder,
                 projection_dim,
                 style_text_embedding_dim,
                 num_patches,
                 **kwargs):
        super().__init__(**kwargs)
        self.content_patch_embedder = content_patch_embedder
        self.style_patch_embedder = style_patch_embedder
        self.content_encoder = content_encoder
        self.style_encoder = style_encoder
        self.cape_layer = cape_layer
        self.refinement_decoder = refinement_decoder
        self.projection_dim = projection_dim
        self.style_text_embedding_dim = style_text_embedding_dim
        self.num_patches = num_patches

        self.combine_features_layer = klayers.Dense(self.projection_dim, name='combine_features')
        self.text_to_style_projection = klayers.Dense(self.num_patches * self.projection_dim,
                                                     name="text_to_style_projection_layer")

    def call(self, inputs, training=False):
        content_img, style_img, style_text_embedding = inputs

        # Patch Embedding
        c_patches = self.content_patch_embedder(content_img)
        s_patches = self.style_patch_embedder(style_img)

        # Transformer Encoding
        c_feats = self.content_encoder(c_patches)
        s_feats = self.style_encoder(s_patches)

        # Reshape for CAPE (Convert sequence to spatial grid)
        # Using patch_size=8
        h = tf.shape(content_img)[1] // 8
        w = tf.shape(content_img)[2] // 8
        c_feats_spatial = tf.reshape(c_feats, [-1, h, w, self.projection_dim])
        s_feats_spatial = tf.reshape(s_feats, [-1, h, w, self.projection_dim])

        # Content-Aware Positional Encoding
        # Pass only the feature tensors; CAPE now handles dimensions dynamically
        combined_features = self.cape_layer([c_feats_spatial, s_feats_spatial])

        # Decoding to Image
        stylized_image = self.refinement_decoder(combined_features)

        # Add dummy text connection to keep the functional graph valid
        text_contribution = tf.reduce_sum(style_text_embedding) * 0.0
        return stylized_image + text_contribution

    def get_config(self):
        config = super().get_config()
    
        config.update({
            "projection_dim": self.projection_dim,
            "style_text_embedding_dim": self.style_text_embedding_dim,
            "num_patches": self.num_patches,
    
            "content_patch_embedder":
                keras.saving.serialize_keras_object(
                    self.content_patch_embedder
                ),
    
            "style_patch_embedder":
                keras.saving.serialize_keras_object(
                    self.style_patch_embedder
                ),
    
            "content_encoder":
                keras.saving.serialize_keras_object(
                    self.content_encoder
                ),
    
            "style_encoder":
                keras.saving.serialize_keras_object(
                    self.style_encoder
                ),
    
            "cape_layer":
                keras.saving.serialize_keras_object(
                    self.cape_layer
                ),
    
            "refinement_decoder":
                keras.saving.serialize_keras_object(
                    self.refinement_decoder
                ),
        })
    
        return config

    @classmethod
    def from_config(cls, config):
    
        config["content_patch_embedder"] = (
            keras.saving.deserialize_keras_object(
                config["content_patch_embedder"]
            )
        )
    
        config["style_patch_embedder"] = (
            keras.saving.deserialize_keras_object(
                config["style_patch_embedder"]
            )
        )
    
        config["content_encoder"] = (
            keras.saving.deserialize_keras_object(
                config["content_encoder"]
            )
        )
    
        config["style_encoder"] = (
            keras.saving.deserialize_keras_object(
                config["style_encoder"]
            )
        )
    
        config["cape_layer"] = (
            keras.saving.deserialize_keras_object(
                config["cape_layer"]
            )
        )
    
        config["refinement_decoder"] = (
            keras.saving.deserialize_keras_object(
                config["refinement_decoder"]
            )
        )
    
        return cls(**config)
