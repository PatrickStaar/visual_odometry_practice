import numpy as np
from models import DepthModel
from conf import conf
from matplotlib import pyplot as p
from util import img_read, depth_read
import json

x = img_read(conf['data_path'][3] + 'rgb/6.jpg')
gt = np.squeeze(depth_read(conf['data_path'][3] + 'depth/6.png'))

depth = DepthModel(input_shape=conf['input_shape'])
depth.model_from_file(conf['depth_model_for_pred'])

depth.load_weights(conf['depth_weights'])
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
