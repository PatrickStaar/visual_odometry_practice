import numpy as np
from depth_model import DepthModel
from conf import conf
from matplotlib import pyplot as p
from util import img_read, depth_read

x = img_read(conf['data_path'][2] + 'rgb/1341847981.358690.png')
gt = np.squeeze(depth_read(conf['data_path'][2] + 'depth/1341847981.358724.png'))

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
