import os
from PIL import Image
from keras.layers import *
from depthmap import Depth
from pose_utils import *
from math import atan2

def conv(inputs, filters, kernel, stride=(1, 1), padding='valid', input_shape=None, name=None):
    return Conv2D(filters=filters,
                  kernel_size=kernel,
                  strides=stride,
                  padding=padding,
                  name=name, )(inputs)


def deconv(inputs, filters, kernel, stride, input_shape=None, name=None):
    return Conv2DTranspose(filters=filters,
                           kernel_size=kernel,
                           strides=stride,
                           padding='same',
                           name=name
                           )(inputs)


def activiate(inputs, type='relu'):
    return Activation(activation=type)(inputs)


def batchnorm(inputs, name=None, axis=-1):
    return BatchNormalization(axis=3, name=name)(inputs)


def pooling(inputs, size, type='max', padding='valid'):
    if type is 'max':
        return MaxPool2D(pool_size=size, padding=padding)(inputs)
    else:
        return AveragePooling2D(pool_size=size, padding=padding)


# depth map smoothness loss
def smoothness(depth):
    def gradient(pred):
        D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        return D_dx, D_dy

    dx, dy = gradient(depth)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)
    smoothness = K.mean(K.abs(dx2)) + K.mean(K.abs(dxdy)) + K.mean(K.abs(dy2))
    return smoothness


# data reading
def img_read(filename, size=(480, 640), aug=False):
    img = np.array(Image.open(filename), dtype=np.float32)
    img = np.resize(img, (size[0], size[1], 3))
    img = img / 255.0
    if aug:
        img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
        img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
        img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
    return img


def depth_read(filename, size=(480, 640), interp=False):
    dep = Depth(filename)
    if interp:
        dep.interpolate()
    depthmap = np.resize(dep.depthmap, (size[0], size[1]))
    depthmap = 1 / depthmap
    depthmap = np.expand_dims(depthmap, axis=-1)
    return depthmap


def pose_read(items):
    assert len(items) >= 7
    vec = [float(i) for i in items]
    pose_mat = transform44(vec)
    return pose_mat


def mat2vec(mat, mode='q'):
    mat = np.reshape(mat, (4, 4))
    t = np.reshape(mat[:3, 3:], (3,))
    rmat = mat[:3, :3]
    if mode == 'q':
        qw = np.sqrt(np.trace(rmat) + 1) / 2
        qx = (rmat[1, 2] - rmat[2, 1]) / (4 * qw)
        qy = (rmat[2, 0] - rmat[0, 2]) / (4 * qw)
        qz = (rmat[0, 1] - rmat[1, 0]) / (4 * qw)
        r = np.reshape([qx, qy, qz, qw], (4,))
    else:
        r = mat2euler(mat)
    vec = np.concatenate([t, r])
    return vec


def mat2euler(mat):
    thetax = atan2(mat[2, 1], mat[2, 2])
    thetay = atan2(-mat[2, 0], np.sqrt(mat[2, 1] ** 2 + mat[2, 2] ** 2))
    thetaz = atan2(mat[1, 0], mat[0, 0])
    return np.reshape([thetaz, thetay, thetax], (3,))
