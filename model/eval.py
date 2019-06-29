import numpy as np
from vo import VO
from matplotlib import pyplot as plt
from models import DepthModel
from conf import conf
from synthesis import syn_model
import cv2
from mpl_toolkits.mplot3d import Axes3D


class Eval:

    def __init__(self, *args, **kwargs):
        self.model = None
        self.depth_model = None

    def load_model(self):
        v = VO(input_shape=(1, 480, 640, 3), mode='test')
        v.model_from_file(conf['end2end_for_pred'])
        v.load_weights([conf['end2end_weights']])
        self.vo = v

    def pred(self, frame0, frame1):
        frame0 = self.img_process(frame0)
        frame1 = self.img_process(frame1)
        d, p = self.vo.model.predict([frame0, frame1])
        d = np.reshape(d, (d.shape[1], d.shape[2]))
        d = 1 / (d + 1e-3)
        d = np.clip(d, 0, 10.0)
        return {'depth': d, 'pose': p, 'frame0': frame0, 'frame1': frame1}

    def depth_estimate(self, frame):

        frame = self.img_process(frame)
        d = self.depth_model.model.predict(frame)
        d = np.reshape(d, (d.shape[1], d.shape[2]))
        # d=1/(d+1e-3)
        # d=np.clip(d,0,15.0)
        return {'monodepth': d}

    def pred_video(self, file):
        frame0 = None
        frame1 = None
        traj = []

        cap = cv2.VideoCapture(file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        writer = cv2.VideoWriter("output.mp4",
                                 cv2.VideoWriter_fourcc("X", "V", "T", "D"),
                                 fps,
                                 (640, 480))

        while cap.isOpened():
            ok, frame = cap.read()

            # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            if not ok:
                break

            frame[:, :, [0, 2]] = frame[:, :, [2, 0]]
            if frame0 is None:
                frame0 = self.img_process(frame)
                continue
            frame1 = self.img_process(frame)
            d, p = self.vo.model.predict([frame0, frame1])
            print('1 pair processed')
            d = np.reshape(d, (d.shape[1], d.shape[2]))
            d = np.clip(d, 0.10, 5.0)
            d = d * 50
            d = np.uint8(d)
            writer.write(d)
            traj.append(np.reshape(p, (6,)))
            frame0 = frame1
        print('video saved')
        cap.release()
        writer.release()

        return {'traj': traj}

    def load_depth_model(self):
        d = DepthModel(input_shape=conf['input_shape'], mode='single')
        d.model_from_file(conf['depth_model_for_pred'])
        d.load_weights(conf['depth_weights'])
        self.depth_model = d

    def get_data(self):
        pass

    def img_process(self, img, size=(480, 640), aug=False):
        img = np.array(img, dtype=np.float32)
        img = np.resize(img, (size[0], size[1], 3))
        img = img / 255.0
        if aug:
            img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
            img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
            img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225

        img = np.expand_dims(img, axis=0)
        return img

    def save(self, output, path):
        plt.imsave(path, output)
        pass

    def visual(self, output):
        if 'traj' in output.keys():

            fig = plt.figure()
            ax = fig.gca(projection='3d')

            traj = output['traj']
            traj = np.array(traj)[:, :3]

            for i in range(1, traj.shape[0]):
                traj[i, :] = traj[i - 1, :] + traj[i, :]

            x, y, z = np.split(traj, 3, axis=-1)
            x = np.squeeze(x, axis=-1)
            y = np.squeeze(y, axis=-1)
            z = np.squeeze(z, axis=-1)
            ax.plot(x, y, z, label='trajectory_gt')
            plt.show()
            return

        if 'monodepth' in output.keys():
            depthmap = output['monodepth']
            plt.figure()
            plt.imshow(depthmap)
            plt.colorbar()
            plt.show()
            return

        if 'pose' in output.keys():
            d = output['depth']
            p = output['pose']
            frame0 = output['frame0']
            frame1 = output['frame1']

            d = np.reshape(d, (1, 480, 640, 1))
            syn = syn_model().predict([frame1, d, p])
            d = np.reshape(d, (480, 640))
            pose = np.reshape(p, (6,))

            fig = plt.figure()
            fig.suptitle('Relative Transform:{}\nRelative Rotation:{}'.format(pose[:3], pose[3:]))
            i0 = plt.subplot(221)
            i0.set_title('I0')
            i0.imshow(frame0.squeeze(axis=0))

            i1 = plt.subplot(222)
            i1.set_title('I1')
            i1.imshow(frame1.squeeze(axis=0))

            isyn = plt.subplot(223)
            isyn.set_title('I0 Estimation from model')
            isyn.imshow(syn.squeeze(axis=0))

            dep = plt.subplot(224)
            dep.set_title('Depthmap')
            dep.imshow(d)

            plt.show()
            return
