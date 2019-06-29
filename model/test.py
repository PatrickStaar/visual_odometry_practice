import numpy as np
from models import DepthModel,OdometryModel
from conf import conf
from matplotlib import pyplot as p
from util import img_read, depth_read, pose_read, mat2vec
from pose_utils import ominus
from PIL import Image
from matplotlib import pyplot as plt
from vo import VO
from synthesis import syn_model

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
    visual = True

    with open(conf['test_path']+'list.txt') as f:
        for l in f.readlines():
            filename=l.strip('\n')
            tests.append(filename)
            gts.append(filename.replace('jpg', 'png'))

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


def end2end():
    v = VO(input_shape=(1, 480, 640, 3), mode='test')
    v.model_from_file(conf['end2end_for_pred'])
    # v.save_as_json(conf['end2end_for_pred'])
    v.load_weights([conf['end2end_weights']])

    #  rgb/1341847981.358690.png  -0.6527 2.6990 1.7395 0.0083 0.8604 -0.5093 -0.0175

    img0 = np.expand_dims(img_read(conf['data_path'][2] + 'rgb/1341847981.322892.png'), axis=0)
    img1 = np.expand_dims(img_read(conf['data_path'][2] + 'rgb/1341847981.390810.png'), axis=0)
    p0 = pose_read([-0.6569, 2.6976, 1.7395, 0.0074, 0.8602, -0.5096, -0.0167])
    p1 = pose_read([-0.6495, 2.7000, 1.7397, 0.0094, 0.8600, -0.5099, -0.0174])

    relative = mat2vec(ominus(p1, p0), mode='e')
    relative = np.expand_dims(relative, axis=0)

    # d,p,s = v.model.predict([img0,img1])
    d, p = v.model.predict([img0, img1])
    d = np.reshape(d, (480, 640))
    d = 1 / (d + 1e-3)

    # s=s.squeeze(axis=0)
    print(p)
    print(relative)

    d = np.reshape(d, (1, 480, 640, 1))
    syn_from_estimation = syn_model().predict([img1, d, p])
    syn_from_gt = syn_model().predict([img1, d, relative])

    visual = True
    if visual:
        plt.figure(1)

        i0 = plt.subplot(142)
        i0.set_title('I0')
        i0.imshow(img0.squeeze(axis=0))

        i1 = plt.subplot(141)
        i1.set_title('I1')
        i1.imshow(img1.squeeze(axis=0))

        isyn = plt.subplot(143)
        isyn.set_title('I0 Estimation from model')
        isyn.imshow(syn_from_estimation.squeeze(axis=0))

        igt = plt.subplot(144)
        igt.set_title('I0 Estimation from gt')
        igt.imshow(syn_from_gt.squeeze(axis=0))

        plt.show()

    img0 = img0.squeeze(axis=0)[40:440, 40:600, :]
    img1 = img1.squeeze(axis=0)[40:440, 40:600, :]
    syn_from_estimation = syn_from_estimation.squeeze(axis=0)[40:440, 40:600, :]
    syn_from_gt = syn_from_gt.squeeze(axis=0)[40:440, 40:600, :]

    # print('t0 t1 mae:',np.mean(np.abs(img0-img1)))
    print('模型输出位姿合成mae:', np.mean(np.abs(img0 - syn_from_estimation)))
    print('标准相对位姿合成mae:', np.mean(np.abs(img0 - syn_from_gt)))


# depth_eval()
end2end()
