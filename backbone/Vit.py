from tensorflow.keras import layers, Model

try:
    from .DropBlock import DropBlock2D
except:
    from DropBlock import DropBlock2D
import tensorflow as tf
import tensorflow_addons as tfa
import math

# image_size = 84
# patch_size = 8
# input_shape = (image_size, image_size, 3)  # input image shape
# patch_ = image_size // patch_size
# num_patches = (patch_) ** 2
# projection_dim = 320
# num_heads = 4
# # Size of the transformer layers
# transformer_units = [
#     projection_dim * 2,
#     projection_dim,
# ]
# transformer_layers = 4
mlp_head_units = [2048, 1024, 512, 64, 32]  #
image_size = 84
patch_size = 8
input_shape = (image_size, image_size, 3)  # input image shape
patch_ = image_size // patch_size
num_patches = (patch_) ** 2
projection_dim = 240
num_heads = 4
# Size of the transformer layers
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 12

def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / 3.1415926) * (x + 0.044715 * tf.pow(x, 3))))

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    #     Override function to avoid error while saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "input_shape": input_shape,
                "patch_size": patch_size,
                "num_patches": num_patches,
                "projection_dim": projection_dim,
                "num_heads": num_heads,
                "transformer_units": transformer_units,
                "transformer_layers": transformer_layers,
                "mlp_head_units": mlp_head_units,
            }
        )
        return config

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        # return patches
        return tf.reshape(patches, [batch_size, -1, patches.shape[-1]])


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    # Override function to avoid error while saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "input_shape": input_shape,
                "patch_size": patch_size,
                "num_patches": num_patches,
                "projection_dim": projection_dim,
                "num_heads": num_heads,
                "transformer_units": transformer_units,
                "transformer_layers": transformer_layers,
                "mlp_head_units": mlp_head_units,
            }
        )
        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        # x = layers.Dense(units, activation=tf.nn.le)(x)
        x = layers.Dense(units, activation=tf.nn.leaky_relu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def Vit(
        input_shape, pooling=False, use_bias=False,
        name="Vit"):
    inputs = layers.Input(shape=input_shape)
    # Create patches
    patches = Patches(patch_size)(inputs)
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = tfa.layers.MultiHeadAttention(
            num_heads=num_heads, head_size=projection_dim, dropout=0.1
        )([x1, x1])
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    # representation = layers.Dropout(0.3)(representation)

    features = tf.keras.layers.Reshape((patch_, patch_, projection_dim))(representation)
    _, r, w, c = features.shape
    min_size = int(min(math.ceil(r / 2), math.ceil(w / 2)))
    dropblock_size = min(min_size, 5)
    features = DropBlock2D(block_size=dropblock_size, keep_prob=1. - 0.2)(features)
    # Add MLP.
    # features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.3)

    # return Keras model.
    return Model(inputs=inputs, outputs=features, name=name)


# model = Vit((84, 84, 3))
# model.summary()
