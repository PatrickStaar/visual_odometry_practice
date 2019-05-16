import numpy as np
from depth_model import DepthModel
from conf import conf
from matplotlib import pyplot as p
from util import img_read, depth_read

x = img_read(conf['data_path'] + 'rgb/1305031102.175304.png')
gt = np.squeeze(depth_read(conf['data_path'] + 'depth/1305031102.160407.png'))

depth = DepthModel(input_shape=conf['input_shape'])
depth.load_weights(conf['weights'])
print('weights loaded')

y = depth.model.predict(np.expand_dims(x, axis=0))
y = np.reshape(y, (480, 640))

p.figure(1)
p.subplot(131)
p.imshow(x)
p.subplot(132)
p.imshow(gt)
p.subplot(133)
p.imshow(y)
p.show()
