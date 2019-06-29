from vo import VO
from util import *
from keras.callbacks import *
from conf import conf
from data_generator import vo_generator
from keras.utils import plot_model
from matplotlib import pyplot as plt
import time

batch = 1
v = VO(input_shape=(batch, 480, 640, 3), mode='train')
v.build(frozen=True,
        separated_weights=True)

v.model.summary()
v.save_as_json(conf['end2end'])
plot_model(v.model, 'visual_odometry.png', show_shapes=True)

v.load_weights([conf['depth_weights'], conf['pose_weights']])
v.compile()

tf_board = TensorBoard(log_dir='./log')
new_weights = 'vo-' + time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time())) + '.h5'
save_check = ModelCheckpoint(filepath='./weights/' + new_weights,
                             monitor='loss',
                             save_best_only=True,
                             save_weights_only=True,
                             mode='auto')

lr_decay = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1, mode='auto')

callbacks = [save_check, lr_decay]  # ,tf_board]

print('start training')

v.model.fit_generator(vo_generator(conf['data_path'],
                                   batch_size=batch),
                      epochs=20,
                      steps_per_epoch=100,
                      verbose=1,
                      callbacks=callbacks)
