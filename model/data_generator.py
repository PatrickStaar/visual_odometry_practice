import os
import numpy as np
from util import depth_read, img_read, pose_read, ominus, mat2vec
from random import randint


def vo_generator(base_dir, batch_size=16):
    '''
    generator for complete vo model training
    by fit_generator()
    '''
    image_list = []
    stamp_list = []
    for b in base_dir:
        with open(b + 'rgb.txt') as img_list:
            for line in img_list:
                if '#' in line:
                    continue
                line = line.rstrip('\n')
                timestamp, rgb = line.split(' ')
                image_list.append(b + rgb)
                stamp_list.append(float(timestamp))

    while True:
        indices = np.random.choice(a=range(len(image_list)), size=batch_size)
        batch_input0 = []
        batch_input1 = []

        for i in indices:
            skip = 1  # randint(1,6)
            if i + skip > len(image_list) - 1:
                t0 = i - skip
                t1 = i
            elif abs(stamp_list[i] - stamp_list[i + skip]) > 10.0:
                t0 = i
                t1 = i - skip
            else:
                t0 = i
                t1 = i + skip

            img0 = img_read(image_list[t0], aug=True)
            img1 = img_read(image_list[t1], aug=True)
            batch_input0.append(img0)
            batch_input1.append(img1)

        batch_input0 = np.array(batch_input0)
        batch_input1 = np.array(batch_input1)

        yield [batch_input0, batch_input1], []


def depth_model_generator(base_dir, batch_size=16):
    '''
    generator for depth prediction model training
    by fit_generator()
    '''
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


def pose_generator(base_dir, batch_size=64):
    image_list = []
    stamp_list = []
    pose_list = []

    for b in base_dir:
        with open(b + 'associated_poses.txt') as img_list:
            for line in img_list:
                line = line.rstrip('\n')
                timestamp, rgb, _, tx, ty, tz, r1, r2, r3, r4 = line.split(' ')
                image_list.append(b + rgb)
                stamp_list.append(float(timestamp))
                pose_list.append(pose_read([tx, ty, tz, r1, r2, r3, r4]))
    indexes = [i for i in range(len(image_list))]
    while True:
        indices = np.random.choice(a=indexes, size=batch_size)
        batch_input0 = []
        batch_input1 = []
        batch_gt = []

        for i in indices:
            skip = 1  # randint(1, 6)
            if i + skip > len(image_list) - 1:
                t0 = i - skip
                t1 = i
            elif abs(stamp_list[i] - stamp_list[i + skip]) > 10.0:
                t0 = i
                t1 = i - skip
            else:
                t0 = i
                t1 = i + skip

            img0 = img_read(image_list[t0], aug=False)
            img1 = img_read(image_list[t1], aug=False)
            batch_input0.append(img0)
            batch_input1.append(img1)
            batch_gt.append(mat2vec(ominus(pose_list[t1], pose_list[t0]), mode='e'))

        batch_input0 = np.array(batch_input0)
        batch_input1 = np.array(batch_input1)
        batch_gt = np.array(batch_gt)

        yield [batch_input0, batch_input1], batch_gt
