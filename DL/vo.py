from depth_model import DepthModel
from odometry import OdometryModel
from keras.layers import Input
from keras import Model


class VO:
    def __init__(self, input_shape):
        self.model = None
        self.input_shape = input_shape
        self.frame0 = None  # 当前帧
        self.frame1 = None  # 前一帧

    def _build(self):
        self.frame0 = Input(shape=self.input_shape)
        self.frame1 = Input(shape=self.input_shape)

        dep = DepthModel(input_tensor=self.frame0)
        odo = OdometryModel(inputs=[self.frame0, self.frame1])

        depthmap = dep.model.output
        pose = odo.model.output

        self.model = Model(inputs=[self.frame0, self.frame1],
                           outputs=[depthmap, pose])

    def losses(self):
        pass
