import tensorflow as tf
import keras
from keras import layers


@keras.utils.register_keras_serializable()
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ffn_units, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim # head dimension
        self.num_heads = num_heads # attention heads
        self.ffn_units = ffn_units # units
        self.dropout_rate = dropout_rate

        # normalizing input before attention (zero mean and unit variance)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        # Multi-Head Attention
        # splits the dimensional embedding into heads
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
        # computing self-attention after normalization
        # self attention with Q,V,K using the same tensor
        attn_output = self.att(query=norm1, value=norm1, key=norm1)
        attn_output = self.dropout1(attn_output, training=training)
        # residual connection that adds original signal back to attention output
        # if the attention finds nothing useful, model can simply pass original signal through
        out1 = inputs + attn_output

        # Feed-Forward Network (FFN)
        norm2 = self.layernorm2(out1) # normalization again before ffn
        ffn_output = self.ffn_layers(norm2)
        ffn_output = self.dropout2(ffn_output, training=training)
        return out1 + ffn_output # second residual connection

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
        # storing the patch size and the dimension of the embedding
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        # flattening the patches from (h, w, c*p*p) to (-1, c*p*p) 
        # -1 means automatic finding of the number of patches
        self.flatten_patches = layers.Reshape((-1, (patch_size * patch_size * 3)))
        # for projecting the flattened patches' numbers into projection_dim numbers
        self.projection = layers.Dense(projection_dim)

    def call(self, images):
        # Extract patches using tf.image.extract_patches
        patches = tf.image.extract_patches(
            images=images,
            # Window size is [batch, height = patch_size, width = patch_size, channels]
            sizes=[1, self.patch_size, self.patch_size, 1],
            # stride equals the patch size so that patches do not overlap
            # moving one patch each time
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            # should ignore incomplete patches at image borders
            padding='VALID',
        )
        # flattening and projecting the patches
        patches = self.flatten_patches(patches)
        # connecting the projection Dense layer to the patches as input
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

        # The main convolutional path for the residual block
        # the number of filters is set from initialization
        # each output pixel looks at a 3x3 neighborhood with added padding to get same output
        self.res_conv1 = layers.SeparableConv2D(filters, kernel_size=3, padding='same', use_bias=False)
        # using batch normalization to stop varying values and ensure stable training
        self.res_bn1 = layers.BatchNormalization()
        # applying to add non linearlity
        self.res_act1 = layers.Activation('relu')

        # the second feature extraction
        self.res_conv2 = layers.SeparableConv2D(filters, kernel_size=3, padding='same', use_bias=False)
        # stabilizing output before residual addition
        self.res_bn2 = layers.BatchNormalization() 

        # shortcut path for residual connection, set to None if channel matching is needed
        self.shortcut_conv = None

        # the Upsampling operation to double the image size
        # bilinear interpolation estimates smooth values, instead of copying pixels
        self.upsample_layer = layers.UpSampling2D(size=2, interpolation='bilinear')

    def build(self, input_shape):
        # creating a 1x1 convolution for the shortcut if input channels don't match the block's output filters
        if input_shape[-1] != self.filters:
            self.shortcut_conv = keras.Sequential([
                # the 1x1 conv changes only the channel 
                layers.Conv2D(self.filters, kernel_size=1, padding='same', use_bias=False),
                layers.BatchNormalization()
            ], name='shortcut_conv_for_channel_matching')
        super().build(input_shape)

    def call(self, inputs):
        # saving the residual
        residual = inputs

        # Connecting the first convolution, then normalization and relu
        x = self.res_conv1(inputs)
        x = self.res_bn1(x)
        x = self.res_act1(x)
        # Connecting the second convolution
        x = self.res_conv2(x)
        x = self.res_bn2(x)
        # applying shortcut transformation if needed
        if self.shortcut_conv is not None:
            # residual shape should match the main branch
            residual = self.shortcut_conv(residual)

        # adding the residual for residual learning, then applying relu
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)
        # performing upsampling
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
        # fixed grid size for intermediate feature map size
        self.target_spatial_size = target_spatial_size 
        self.projection_dim = projection_dim # Transformer embedding dimension
        # 1×1 convolution to mix feature channels
        self.conv1x1 = layers.Conv2D(projection_dim, kernel_size=1, activation='gelu', name='cape_conv1x1')

    def call(self, image_features, output_sequence_length):
        batch_size = tf.shape(image_features)[0] # extracting the batch
        # resizing to ensure the model analyzes spatial context at a consistent scale
        pooled_features = tf.image.resize(
            image_features,
            size=(self.target_spatial_size, self.target_spatial_size),
            method=tf.image.ResizeMethod.BILINEAR # standardizing spatial sizes
        )
        cape_representation = self.conv1x1(pooled_features) # generating content aware embeddings
        # computing square root of output_sequence_length as square grid side
        output_spatial_side = tf.cast(tf.sqrt(tf.cast(output_sequence_length, tf.float32)), tf.int32)
        # ensuring that the resulting square grid is equal to output_sequence_length
        tf.Assert(tf.equal(output_spatial_side * output_spatial_side, output_sequence_length),
                  ["Input output_sequence_length must be a perfect square for reshaping to a 2D grid."])
        # resizing again to match the transformer's grid token
        rescaled_cape = tf.image.resize(
            cape_representation,
            size=(output_spatial_side, output_spatial_side),
            method=tf.image.ResizeMethod.BILINEAR
        )
        # flattening into tokens
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
        
