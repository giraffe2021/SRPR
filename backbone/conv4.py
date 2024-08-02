import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, Model, Sequential, optimizers, metrics
try:
    from .DropBlock import DropBlock2D
except:
    from DropBlock import DropBlock2D


def conv4_net(input_shape=None, pooling=None, use_bias=True):
    if input_shape is None:
        input_shape = (84, 84, 3)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1), name='conv1_pad'))
    model.add(
        tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same',
                               input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(
        tf.keras.layers.Conv2D(160, (3, 3), kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(
        tf.keras.layers.Conv2D(320, (3, 3), kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(DropBlock2D(block_size=5, keep_prob=0.8))
    model.add(
        tf.keras.layers.Conv2D(640, (3, 3), kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(DropBlock2D(block_size=5, keep_prob=0.8))
    # model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    if pooling == 'avg':
        model.add(tf.keras.layers.GlobalAveragePooling2D())
    elif pooling == 'max':
        model.add(tf.keras.layers.GlobalMaxPooling2D())
    x = tf.keras.layers.Input(shape=input_shape)
    out = model(x)
    return tf.keras.Model(x, out)
