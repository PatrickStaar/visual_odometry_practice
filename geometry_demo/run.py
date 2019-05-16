import cv2
import numpy as np
from vo import vo
from cfg import Config
from frame import Frame
from matplotlib import pyplot as plt


conf = Config('./cfg/freiburg.json')

vio = vo(conf)
outfile = './out.txt'
x=1.33
y=0.64
z=1.65
# imgs = []
# deps=[]
#
# f=plt.figure()
#
# for i in [1,2]:
#     imgs.append(cv2.imread('./data/'+str(i)+'.png'))
#     deps.append(cv2.imread('./data/'+str(i)+'d.png', -1))
#
# for i in range(len(imgs)):
#     id = int(i)
#     fr = Frame(id=id,
#                img=imgs[i],
#                depth=deps[i],
#                camera=vio.camera
#                )
#     vio.add_frame(fr,visual=True)
#     if vio.status == -1:
#         break
#     if vio.t is not None:
#         t=vio.t.T
#         print('translation:', t[0,0], t[0,1], t[0,2])
#         print('rotation:\n',vio.R)
#

with open(outfile, 'w') as out:
    with open(conf.path + 'associated.txt') as img_list:
        indx = 0
        id = -1
        for line in img_list:
            if indx == 10:
                break
            line=line.rstrip('\n')
            timestamp, rgb, t, depth = line.split(' ')
            timestamp = float(timestamp)
            id += 1
            fr = Frame(id=id,
                       stamp=timestamp,
                       pose=None,
                       camera=vio.camera,
                       img=cv2.imread(conf.path + rgb),
                       depth=cv2.imread(conf.path + depth,-1))

            vio.add_frame(fr,visual=True)
            if vio.status == -1:
                break
            if vio.t is not None:
                t=vio.t.T
                x,y,z=t[0, 0] +x , t[0, 1] +y, t[0, 2] + z
                print('{:.4f} {:>,.4f} {:.4f} {:.4f}'.format(fr.timestamp, x, y, z), file=out)
            indx += 1
            #print('processed:', id)