import tensorflow as tf
import keras
from keras import layers
# from tensorflow import keras
# from tensorflow.keras import layers


@keras.utils.register_keras_serializable()
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ffn_units, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_units = ffn_units
        self.dropout_rate = dropout_rate

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.dropout1 = layers.Dropout(dropout_rate)

        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn_layers = keras.Sequential(
            [
                layers.Dense(ffn_units, activation="gelu"),
                layers.Dense(embed_dim),
            ]
        )
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        norm1 = self.layernorm1(inputs)
        attn_output = self.att(query=norm1, value=norm1, key=norm1)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output

        norm2 = self.layernorm2(out1)
        ffn_output = self.ffn_layers(norm2)
        ffn_output = self.dropout2(ffn_output, training=training)
        return out1 + ffn_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ffn_units": self.ffn_units,
            "dropout_rate": self.dropout_rate,
        })
        return config
        

@keras.utils.register_keras_serializable()
class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        # Reshape layer to flatten the patches from (h, w, c*p*p) to (-1, c*p*p)
        # Assuming 3 channels for images
        self.flatten_patches = layers.Reshape((-1, (patch_size * patch_size * 3)))
        self.projection = layers.Dense(projection_dim)

    def call(self, images):
        # Extract patches using tf.image.extract_patches
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patches = self.flatten_patches(patches)
        projected_patches = self.projection(patches)
        return projected_patches

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "projection_dim": self.projection_dim,
        })
        return config
        

@keras.utils.register_keras_serializable()
class DecoderBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

        # Main convolutional path for the residual block
        # Using SeparableConv2D as per "Xception-like" principles
        self.res_conv1 = layers.SeparableConv2D(filters, kernel_size=3, padding='same', use_bias=False)
        self.res_bn1 = layers.BatchNormalization()
        self.res_act1 = layers.Activation('relu') # ReLU is a common choice for decoder activations

        self.res_conv2 = layers.SeparableConv2D(filters, kernel_size=3, padding='same', use_bias=False)
        self.res_bn2 = layers.BatchNormalization()

        # Shortcut path for residual connection, initialized in build if channel matching is needed
        self.shortcut_conv = None

        # Upsampling operation, as specified to be part of each stage
        self.upsample_layer = layers.UpSampling2D(size=2, interpolation='bilinear')

    def build(self, input_shape):
        # Create a 1x1 convolution for the shortcut if input channels don't match the block's output filters
        if input_shape[-1] != self.filters:
            self.shortcut_conv = keras.Sequential([
                layers.Conv2D(self.filters, kernel_size=1, padding='same', use_bias=False),
                layers.BatchNormalization()
            ], name='shortcut_conv_for_channel_matching')
        super().build(input_shape)

    def call(self, inputs):
        # Store inputs for the residual connection
        residual = inputs

        # Main convolutional path
        x = self.res_conv1(inputs)
        x = self.res_bn1(x)
        x = self.res_act1(x)

        x = self.res_conv2(x)
        x = self.res_bn2(x)

        # Apply shortcut transformation if necessary
        if self.shortcut_conv is not None:
            residual = self.shortcut_conv(residual)

        # Add residual and main paths, then activate
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x) # Final activation after residual addition

        # Perform upsampling as the last step of the stage
        x = self.upsample_layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config
        

@keras.utils.register_keras_serializable()
class ContentAwarePositionalEncoding(layers.Layer):
    def __init__(self, target_spatial_size, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.target_spatial_size = target_spatial_size
        self.projection_dim = projection_dim
        self.conv1x1 = layers.Conv2D(projection_dim, kernel_size=1, activation='gelu', name='cape_conv1x1')

    def call(self, image_features, output_sequence_length):
        batch_size = tf.shape(image_features)[0]
        pooled_features = tf.image.resize(
            image_features,
            size=(self.target_spatial_size, self.target_spatial_size),
            method=tf.image.ResizeMethod.BILINEAR
        )
        cape_representation = self.conv1x1(pooled_features)
        output_spatial_side = tf.cast(tf.sqrt(tf.cast(output_sequence_length, tf.float32)), tf.int32)
        tf.Assert(tf.equal(output_spatial_side * output_spatial_side, output_sequence_length),
                  ["Input output_sequence_length must be a perfect square for reshaping to a 2D grid."])

        rescaled_cape = tf.image.resize(
            cape_representation,
            size=(output_spatial_side, output_spatial_side),
            method=tf.image.ResizeMethod.BILINEAR
        )
        final_cape_encoding = tf.reshape(
            rescaled_cape,
            (batch_size, output_sequence_length, self.projection_dim)
        )
        return final_cape_encoding

    def get_config(self):
        config = super().get_config()
        config.update({
            "target_spatial_size": self.target_spatial_size,
            "projection_dim": self.projection_dim,
        })
        return config
        
