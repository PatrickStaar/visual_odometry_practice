from models import *
from conf import conf
from keras.losses import mean_squared_error, mean_absolute_error
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from data_generator import pose_generator
from keras.callbacks import TensorBoard

batchsize = 1

vo = OdometryModel(input_shape=(batchsize, 480, 640, 3), mode='supervised')

# Visual
vo.model.summary()
# save_model(conf['posenet'],vo.model)

# plot_model(vo.model, to_file='./odometry.png')
vo.load_weights(conf['pose_weights'])

vo.model.compile(optimizer='adam',
                 loss=mean_absolute_error)
print('compiled')

save_check = ModelCheckpoint(filepath='./weights/pose_6.6.h5',
                             monitor='loss',
                             save_best_only=True,
                             save_weights_only=True,
                             mode='auto')

lr_decay = ReduceLROnPlateau(monitor='loss',
                             factor=0.5,
                             patience=2,
                             min_delta=1e-4,
                             cooldown=1)

tf_board = TensorBoard(log_dir='./log')

callbacks = [save_check, lr_decay, tf_board]
print('start training')
vo.model.fit_generator(pose_generator(conf['data_path'],
                                      batch_size=batchsize),
                       epochs=10,
                       steps_per_epoch=100,
                       verbose=1,
                       callbacks=callbacks)
