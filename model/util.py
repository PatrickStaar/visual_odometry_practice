import os
import cv2
from PIL import Image
from keras.layers import *
from depthmap import Depth


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


def post_process(x):
    # flatten the transform matrix
    x = x[:, :3, :]
    x = K.reshape(x, (-1, 12))
    return x


# data reading

def img_read(filename):
    img = np.array(Image.open(filename), dtype=np.float32)
    img = img / 255.0
    # img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
    # img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
    # img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
    return img


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    dep = Depth(filename)
    # dep.interpolate()
    # depth_png = np.array(Image.open(filename), dtype=int)
    # # make sure we have a proper 16bit depth map here.. not 8bit!
    # assert (np.max(depth_png) > 255)
    #
    # depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.

    depthmap = 1 / dep.depthmap
    depthmap = np.expand_dims(depthmap, axis=-1)
    return depthmap


def paths_generator(directory, list='list.txt'):
    files = []
    with open(os.path.join(directory, list), 'r') as t:
        items = t.readlines()
        for i in items:
            files.append(os.path.join(directory, i.strip()))
    return files


def data_generator(base_dir, batch_size=64):
    image_list = []
    depth_list = []
    for b in base_dir:
        with open(b + 'associated.txt') as img_list:
            for line in img_list:
                line = line.rstrip('\n')
                timestamp, rgb, t, depth = line.split(' ')
                image_list.append(b + rgb)
                depth_list.append(b + depth)
    indexes = [i for i in range(len(image_list))]
    while True:
        input_paths = np.random.choice(a=indexes, size=batch_size)
        batch_input = []
        batch_gt4 = []
        batch_gt3 = []
        batch_gt2 = []
        batch_gt1 = []

        for i in input_paths:
            batch_input.append(img_read(image_list[i]))
            depthmap = depth_read(depth_list[i])
            batch_gt4.append(depthmap)
            height = depthmap.shape[0]
            width = depthmap.shape[1]
            batch_gt3.append(np.resize(depthmap, [int(height / 2), int(width / 2), 1]))
            batch_gt2.append(np.resize(depthmap, [int(height / 4), int(width / 4), 1]))
            batch_gt1.append(np.resize(depthmap, [int(height / 8), int(width / 8), 1]))

        batch_input = np.array(batch_input)
        batch_gt4 = np.array(batch_gt4)
        batch_gt3 = np.array(batch_gt3)
        batch_gt2 = np.array(batch_gt2)
        batch_gt1 = np.array(batch_gt1)

        yield batch_input, [batch_gt4, batch_gt3, batch_gt2, batch_gt1]


def data_total(train_path, gt_path):
    train_files = paths_generator(train_path)
    gt_files = paths_generator(gt_path)

    train_img = np.array([img_read(i) for i in train_files])
    gt_img = np.array([depth_read(i) for i in gt_files])

    return train_img, gt_img


def data_tum(base_dir, multi_losses=False):
    train_img = []
    depth_gt = []
    gt3 = []
    gt2 = []
    gt1 = []
    for b in base_dir:
        with open(b + 'associated.txt') as img_list:
            for line in img_list:
                line = line.rstrip('\n')
                timestamp, rgb, t, depth = line.split(' ')
                # timestamp = float(timestamp)
                train_img.append(img_read(b + rgb))
                depthmap = depth_read(b + depth)
                depth_gt.append(depthmap)
                height = depthmap.shape[0]
                width = depthmap.shape[1]
                if multi_losses:
                    gt3.append(np.resize(depthmap, [int(height / 2), int(width / 2), 1]))
                    gt2.append(np.resize(depthmap, [int(height / 4), int(width / 4), 1]))
                    gt1.append(np.resize(depthmap, [int(height / 8), int(width / 8), 1]))

    train_img = np.array(train_img)
    depth_gt = np.array(depth_gt)
    gt3 = np.array(gt3)
    gt2 = np.array(gt2)
    gt1 = np.array(gt1)
    return train_img, depth_gt, gt3, gt2, gt1
