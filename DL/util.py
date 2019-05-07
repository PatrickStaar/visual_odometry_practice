import os
from PIL import Image
from keras.layers import *


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


def res_identity_block(inputs, output_channels, stage, block):
    '''
    construction:
    kernel  stride  padding
     1x1     1x1     valid
     3x3     1x1     same
     1x1     1x1     valid

    '''
    origine = inputs
    name = 'res{}{}_branch'.format(stage, block)
    bn_name = 'bn{}{}_branch'.format(stage, block)

    res1 = conv(inputs, output_channels[0], kernel=(1, 1), name=name + '2a')
    res1 = batchnorm(res1, name=bn_name + 'bn' + '2a')
    res1 = activiate(res1)

    res2 = conv(res1, output_channels[1], kernel=(3, 3), padding='same', name=name + '2b')
    res2 = batchnorm(res2, name=bn_name + '2b')
    res2 = activiate(res2)

    res3 = conv(res2, output_channels[2], kernel=(1, 1), name=name + '2c')
    res3 = batchnorm(res3, name=bn_name + '2c')
    res3 = activiate(res3)

    output = Add()([res3, origine])
    output = activiate(output)

    return output


def res_conv_block(inputs, output_channels, stage, block, stride=(2, 2)):
    '''
    construction:
    kernel  padding
     1x1     valid
     3x3     same
     1x1     valid

    '''
    origine = inputs
    name = 'res{}{}_branch'.format(stage, block)
    bn_name = 'bn{}{}_branch'.format(stage, block)

    res1 = conv(inputs, output_channels[0], kernel=(1, 1), stride=stride, name=name + '2a')
    res1 = batchnorm(res1, name=bn_name + '2a')
    res1 = activiate(res1)

    res2 = conv(res1, output_channels[1], kernel=(3, 3), padding='same', name=name + '2b')
    res2 = batchnorm(res2, name=bn_name + '2b')
    res2 = activiate(res2)

    res3 = conv(res2, output_channels[2], kernel=(1, 1), name=name + '2c')
    res3 = batchnorm(res3, name=bn_name + '2c')
    res3 = activiate(res3)

    shortcut = conv(origine, filters=output_channels[2], kernel=(1, 1), stride=stride, name=name + '1')
    shortcut = batchnorm(shortcut, name=bn_name + '1')

    output = Add()([res3, shortcut])
    output = activiate(output)

    return output


def decoder_block(input1, input2, output_channels, stage, block='a'):
    # name='deconv{}{}_branch'.format(stage,block)
    # bn_name = 'de_bn{}{}_branch'.format(stage,block)
    x = deconv(input1, output_channels[0], kernel=(4, 4), stride=(2, 2))
    if input2 is not None: # tf.Tensor 不能直接作为bool
        x = Add()([x, input2])

    return x


# data reading

def img_read(filename):
    img =np.array(Image.open(filename),dtype=np.float32)
    img = img / 255.0
    img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
    img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
    img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
    return img


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = 1000.
    depth/=1
    depth=np.expand_dims(depth,axis=-1)
    return depth


def paths_generator(directory, list='list.txt'):
    files=[]
    with open(os.path.join(directory, list), 'r') as t:
        items = t.readlines()
        for i in items:
            files.append(os.path.join(directory, i.strip()))
    return files


def data_generator(paths, batch_size=64):
    while True:
        input_paths = np.random.choice(a=paths, size=batch_size)
        batch_input = []
        batch_gt =[]

        for input_path in input_paths:
            gt_path=input_path.replace('sync_image_', 'sync_groundtruth_depth_').replace('images', 'groundtruth_depth')

            batch_input.append(img_read(input_path))
            batch_gt.append(depth_read(gt_path))

        batch_input= np.array(batch_input)
        batch_gt = np.array(batch_gt)

        yield (batch_input, batch_gt)


def data_total(train_path, gt_path):
    train_files=paths_generator(train_path)
    gt_files=paths_generator(gt_path)

    train_img =np.array([img_read(i) for i in train_files])
    gt_img = np.array([depth_read(i) for i in gt_files])

    return train_img, gt_img
