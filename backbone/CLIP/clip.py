import tensorflow as tf

from .resnet import ModifiedResNet
from .visual_transformer import VisualTransformer


def CLIP_ResNet(input_shape=(224, 224, 3), name="CLIP_ResNet", **kwargs):
    img_input = tf.keras.layers.Input(shape=input_shape)
    encoder = ModifiedResNet(
        layers=(3, 4, 6, 3),
        output_dim=1024,
        heads=32,
        input_resolution=224,
        width=64,
        name="visual"
    )
    encoder.build([None, *input_shape])
    encoder.load_weights("backbone/CLIP/CLIP_RN50.h5")
    x = encoder(img_input)
    x = tf.keras.layers.Reshape([1, 1, -1])(x)
    model = tf.keras.Model(img_input, x, name=name)
    return model


def CLIP_VIT(input_shape=(224, 224, 3), name="CLIP_VIT", **kwargs):
    img_input = tf.keras.layers.Input(shape=input_shape)
    encoder = VisualTransformer(
        input_resolution=224,
        patch_size=14,
        width=1024,
        layers=24,
        heads=16,
        output_dim=768,
        name="visual"
    )
    encoder.build([None, *input_shape])
    encoder.load_weights("backbone/CLIP/ViT-L-14.h5")
    x = encoder(img_input)
    x = tf.keras.layers.Reshape([1, 1, -1])(x)
    model = tf.keras.Model(img_input, x, name=name)
    return model
