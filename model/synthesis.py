import numpy as np
from keras.layers import Input, Lambda
from util import *
from projective import *
from keras.models import Model
from conf import conf
from matplotlib import pyplot as plt


def syn(inputs):
    frame1, depthmap, mat = inputs
    intrinsic = get_intrinsics(conf['intrinsics'])
    return synthesis(frame1, depthmap, mat, (1, 480, 640, 1), intrinsic)


# frame0 = Input(batch_shape=(1, 480, 640, 3))
def syn_model():
    frame1 = Input(batch_shape=(1, 480, 640, 3))
    depthmap = Input(batch_shape=(1, 480, 640, 1))
    pose = Input(batch_shape=(1, 6))

    mat = Lambda(vec2mat, name='euler2mat')(pose)
    synthesis = Lambda(syn)([frame1, depthmap, mat])

    model = Model(inputs=[frame1, depthmap, pose],
                  outputs=[synthesis])
    return model
