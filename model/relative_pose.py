from pose_utils import *
from conf import conf
import numpy as np

traj = read_trajectory(conf['data_path'][0] + 'groundtruth.txt')

stamps = [i for i in traj.keys()]
stamps.sort()

t0 = stamps[0]
t1 = stamps[1]
t2 = stamps[2]

pose10 = ominus(traj[t0], traj[t1])
pose21 = ominus(traj[t1], traj[t2])
pose20 = ominus(traj[t0], traj[t2])
x = np.matmul(pose10, pose21)
print('s')
