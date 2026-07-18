import tensorflow as tf
import keras
from keras import layers as klayers

from layers import DecoderBlock

@keras.utils.register_keras_serializable()
class RefinementDecoder(keras.Model):
    def __init__(self, projection_dim, output_image_size, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.output_image_size = output_image_size
        filter_configs = [projection_dim, projection_dim // 2, projection_dim // 4]
        filter_configs = [max(f, 32) for f in filter_configs]
        self.decoder_blocks = []
        self.decoder_blocks.append(DecoderBlock(filter_configs[0], name="decoder_block_0"))
        self.decoder_blocks.append(DecoderBlock(filter_configs[1], name="decoder_block_1"))
        self.decoder_blocks.append(DecoderBlock(filter_configs[2], name="decoder_block_2"))
        self.final_conv = klayers.Conv2D(3, kernel_size=3, padding='same', activation='sigmoid', name='output_image_conv')

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.shape(inputs)[1]
        spatial_side = tf.cast(tf.sqrt(tf.cast(sequence_length, tf.float32)), tf.int32)
        tf.Assert(tf.equal(spatial_side * spatial_side, sequence_length),
                  ["Input sequence_length must be a perfect square for reshaping to 2D."])
        x = tf.reshape(inputs, (batch_size, spatial_side, spatial_side, self.projection_dim))
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

    def call(self, content_img, style_img=None, style_text_embedding=None, training=False):
        content_patches = self.content_patch_embedder(content_img)
        output_sequence_length = tf.shape(content_patches)[1]
        content_cape_encoding = self.cape_layer(content_img, output_sequence_length)
        content_patches_with_cape = content_patches + content_cape_encoding
        encoded_content = self.content_encoder(content_patches_with_cape)

        if style_img is not None:
            style_patches = self.style_patch_embedder(style_img)
            style_cape_encoding = self.cape_layer(style_img, output_sequence_length)
            style_patches_with_cape = style_patches + style_cape_encoding
            encoded_style = self.style_encoder(style_patches_with_cape)
        else:
            target_batch_size = tf.shape(content_img)[0]
            tiled_style_text_embedding = tf.tile(style_text_embedding, [target_batch_size, 1])
            projected_text_style = self.text_to_style_projection(tiled_style_text_embedding)
            encoded_style = tf.reshape(projected_text_style, [target_batch_size, self.num_patches, self.projection_dim])

        decoder_input = tf.concat([encoded_content, encoded_style], axis=-1)
        combined_features = self.combine_features_layer(decoder_input)
        generated_image = self.refinement_decoder(combined_features)
        return generated_image

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
