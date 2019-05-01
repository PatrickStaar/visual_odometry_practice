from util import depth_read
from matplotlib import pyplot as plt
import numpy as np

dep=depth_read('depth2.png')
dep=1/dep
plt.figure()
plt.imshow(np.reshape(dep,(dep.shape[0],dep.shape[1])))
plt.show()