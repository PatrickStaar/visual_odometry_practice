import numpy as np
from models import DepthModel,OdometryModel
from conf import conf
from matplotlib import pyplot as p
from util import img_read, depth_read
from PIL import Image


def rmse(pred,gt):
    pred=np.reshape(pred,(1,-1))
    gt = np.reshape(gt,(1,-1))
    squared_error=np.square(pred-gt)
    rmse=np.sqrt(np.mean(squared_error))
    return rmse

def pose_eval():
    posenet=OdometryModel(mode='supervised')
    posenet.model_from_file(conf['posenet'])
    posenet.load_weights()

def depth_eval():
    depth = DepthModel(input_shape=conf['input_shape'],mode='single')
    depth.model_from_file(conf['depth_model_for_pred'])
    depth.load_weights(conf['depth_weights'])
    print('weights loaded')
    tests = []
    gts = []
    measure = []
    save = False
    visual = False

    with open(conf['test_path']+'list.txt') as f:
        for l in f.readlines():
            filename=l.strip('\n')
            tests.append(filename)
            gts.append(filename.replace('color','depth'))

    for i in range(len(tests)):
        # Arrage inputs and groundtruth
        gt = depth_read(conf['test_path']+gts[i],interp=True)
        gt = np.squeeze(gt)
        img = img_read(conf['test_path']+tests[i])
        pred = depth.model.predict(np.expand_dims(img, axis=0))
        pred = np.resize(pred,(480,640))

        measure.append(rmse(pred,gt))

        # Visual
        if visual:
            p.figure(1)
            p.subplot(131)
            p.imshow(img)
            p.subplot(132)
            p.imshow(gt)
            p.subplot(133)
            p.imshow(pred)
            p.show()

        # Save as png 16 bit
        if save:
            dmap = pred*5000
            dmap = dmap.astype(np.int16)
            p.imsave(tests[i].replace('color','depth_estimated'),dmap)

    print(np.mean(np.asarray(measure)))