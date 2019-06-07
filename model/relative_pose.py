from pose_utils import *
from conf import conf
import numpy as np
from util import mat2vec

traj = read_trajectory(conf['data_path'][0] + 'groundtruth.txt')

stamps = [i for i in traj.keys()]
stamps.sort()

t0 = stamps[0]
t1 = stamps[1]
t2 = stamps[2]

pose10 = ominus(traj[t0], traj[t1])
pose21 = ominus(traj[t1], traj[t2])
pose20 = ominus(traj[t0], traj[t2])
xx = np.matmul(pose10, pose21)
v = mat2vec(pose20, mode='e')
# z=transform44(y)

z, y, x = v[3:]

# Expand to 1 x 1 x 1
z = np.expand_dims(np.expand_dims(np.expand_dims(z, -1), -1), -1)
y = np.expand_dims(np.expand_dims(np.expand_dims(y, -1), -1), -1)
x = np.expand_dims(np.expand_dims(np.expand_dims(x, -1), -1), -1)

zeros = np.zeros_like(z)
ones = np.ones_like(z)

cosz = np.cos(z)
sinz = np.sin(z)
rotz_1 = np.concatenate([cosz, -sinz, zeros], axis=2)
rotz_2 = np.concatenate([sinz, cosz, zeros], axis=2)
rotz_3 = np.concatenate([zeros, zeros, ones], axis=2)
zmat = np.concatenate([rotz_1, rotz_2, rotz_3], axis=1)

cosy = np.cos(y)
siny = np.sin(y)
roty_1 = np.concatenate([cosy, zeros, siny], axis=2)
roty_2 = np.concatenate([zeros, ones, zeros], axis=2)
roty_3 = np.concatenate([-siny, zeros, cosy], axis=2)
ymat = np.concatenate([roty_1, roty_2, roty_3], axis=1)

cosx = np.cos(x)
sinx = np.sin(x)
rotx_1 = np.concatenate([ones, zeros, zeros], axis=2)
rotx_2 = np.concatenate([zeros, cosx, -sinx], axis=2)
rotx_3 = np.concatenate([zeros, sinx, cosx], axis=2)
xmat = np.concatenate([rotx_1, rotx_2, rotx_3], axis=1)

rotMat = np.dot(np.dot(np.squeeze(xmat, 0), np.squeeze(ymat, 0)), np.squeeze(zmat, 0))



print('s')

