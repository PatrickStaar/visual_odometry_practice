import numpy as np
from models import DepthModel,OdometryModel
from conf import conf
from matplotlib import pyplot as p
from util import img_read, depth_read, pose_read, mat2vec
from pose_utils import ominus
from PIL import Image
from matplotlib import pyplot as plt


def rmse(pred,gt):
    pred=np.reshape(pred,(1,-1))
    gt = np.reshape(gt,(1,-1))
    squared_error=np.square(pred-gt)
    rmse=np.sqrt(np.mean(squared_error))
    return rmse

def pose_eval():
    posenet=OdometryModel(mode='supervised')
    posenet.model_from_file(conf['posenet'])
    posenet.load_weights(conf['pose_weights'])



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

# def end2end():
#     v = VO(input_shape=(1, 480, 640, 3), mode='train')
#
#     img0=np.expand_dims(img_read(conf['data_path'][0]+'rgb/1305031122.715208.png'),axis=0)
#     img1=np.expand_dims(img_read(conf['data_path'][0]+'rgb/1305031122.751298.png'),axis=0)
#     p0=pose_read([1.2349, 0.5621, 1.5526, 0.6350, 0.6658, -0.2987, -0.2534])
#     p1=pose_read([ 1.2469, 0.5606, 1.5446, 0.6346, 0.6668, -0.2990, -0.2515])
#
#     relative=mat2vec(ominus(p1,p0),mode='e')
#     relative=np.expand_dims(relative,axis=0)
#
#     d,p,s = v.model.predict([img0,img1])
#     d=np.reshape(d,(480,640))
#     s=s.squeeze(axis=0)
#     print(p)
#     print(relative)
#
#     d=np.reshape(d,(1,480,640,1))
#     ss=syn_model().predict([img1, d, relative])
#
#     visual=True
#     if visual:
#         plt.figure(1)
#         plt.subplot(141)
#         plt.imshow(img1.squeeze(axis=0))
#         plt.subplot(142)
#         plt.imshow(img0.squeeze(axis=0))
#         plt.subplot(143)
#         plt.imshow(ss.squeeze(axis=0))
#         plt.subplot(144)
#         plt.imshow(s)
#         plt.show()
#
#     img0=img0.squeeze(axis=0)[40:440,40:600,:]
#     img1=img1.squeeze(axis=0)[40:440,40:600,:]
#     ss=ss.squeeze(axis=0)[40:440,40:600,:]
#
#     print(np.mean(np.abs(img0-img1)))
#     print(np.mean(np.abs(img0-ss)))
