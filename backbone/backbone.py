from functools import partial

import tensorflow as tf

from .CLIP import CLIP_ResNet, CLIP_VIT
from .Vit import Vit
from .conv4 import conv4_net
from .model_provider import resnet50
from .resnet18 import ResNet18, ResNet12
from .resnet_12 import ResNet_12
from .wrn_28_10 import wrn_28_10

# self supervised training
back_bone_dict = {"resnet18": ResNet18, "conv4": conv4_net, "wrn_28_10": wrn_28_10, "resnet12": ResNet12,
                  # "resnet50": partial(tf.keras.applications.ResNet50, include_top=False, weights=None),
                  "resnet50": resnet50,
                  "MobileNetV2_0_35": partial(tf.keras.applications.MobileNetV2, alpha=0.35, include_top=False,
                                              weights='imagenet'),
                  "resnet_12": ResNet_12,
                  "Vit": Vit,
                  "CLIP_ResNet": CLIP_ResNet,
                  "CLIP_VIT": CLIP_VIT,
                  }


class Backbone:
    def __init__(self, backbone="conv4", input_shape=(84, 84, 3), pooling='avg', use_bias=True, name=None):
        if name is not None:
            self.encoder = back_bone_dict[backbone](input_shape=input_shape, pooling=pooling,
                                                    name=name)
        else:
            self.encoder = back_bone_dict[backbone](input_shape=input_shape, pooling=pooling, )

    def load_weights(self, path):
        self.encoder.load_weights(path)

    def get_model(self, *args, **kwargs):
        return self.encoder


if __name__ == '__main__':
    encoder = Backbone()
    encoder.load_weights("/data/giraffe/0_FSL/FSL/ckpts/pretrain/export/encoder.h5")
    model = encoder.get_model()
    model.summary()
