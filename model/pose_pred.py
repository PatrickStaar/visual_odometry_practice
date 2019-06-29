from models import OdometryModel
from conf import conf
from util import *

vo = OdometryModel(input_shape=(1, 480, 640, 3), mode='supervised')
vo.build()

# Visual
# o.model.summary()
# plot_model(vo.model, to_file='./odometry.png')
vo.load_weights(conf['pose_weights'])

img0 = np.expand_dims(img_read(conf['data_path'][2] + 'rgb/1341847981.054711.png'), axis=0)
img1 = np.expand_dims(img_read(conf['data_path'][2] + 'rgb/1341847981.090715.png'), axis=0)
pose0 = pose_read([-0.6736, 2.6942, 1.7374, -0.0001, 0.8615, -0.5075, -0.0174])
pose1 = pose_read([-0.6717, 2.6947, 1.7374, 0.0007, 0.8608, -0.5086, -0.0185])
relative = mat2vec(ominus(pose0, pose1), mode='e')
prediction = np.squeeze(vo.model.predict([img0, img1]), axis=0)
error = np.mean(np.abs(relative - prediction))
print(relative)
print('*****************')
print(prediction)
print(error)
