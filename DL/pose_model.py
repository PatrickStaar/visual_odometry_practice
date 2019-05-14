import numpy as np
from keras.applications import VGG16
from keras import Model
from keras.layers import *
from keras.callbacks import *


class PoseModel:
    def __init__(self):
        self.inputs = None
        self.model = None
        self.callbacks = None
        self.params = None
        self.outputs = None
        self.input_shape = None

    def build(self):
        encoder = VGG16(include_top=False,  # 是否保留全连接层
                        weights='imagenet',  # None表示随机初始化
                        input_shape=self.input_shape)

        encoded = encoder.output  # 获取resnet的输出

        x=Dense(units=512,activation='relu')(encoded)
        x=Dense(units=512,activation='relu')(x)
        pose = Dense(units=6, activation='softmax')



    def model_from_file(self, model_file, weights_file=None):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def get_inputs(self):
        pass