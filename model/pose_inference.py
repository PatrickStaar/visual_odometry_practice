from models import OdometryModel
from conf import conf
from keras.losses import mean_squared_error
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from util import pose_generator

vo = OdometryModel(input_shape=(10, 480, 640, 3), mode='supervised')

# Visual
vo.model.summary()
# plot_model(vo.model, to_file='./odometry.png')
vo.load_weights(conf['vgg'], vgg_only=True)

vo.model.compile(optimizer='adam',
                 loss=mean_squared_error)
print('compiled')

# 一次性导入数据
# x, gt4,gt3,gt2,gt1 = data_tum(conf['data_path'],multi_losse                    s=True)
# print('dataset loaded')

save_check = ModelCheckpoint(filepath='./weights/best_pose.h5',
                             monitor='loss',
                             save_best_only=True,
                             save_weights_only=True,
                             mode='auto')

lr_decay = ReduceLROnPlateau(monitor='loss',
                             factor=0.1,
                             patience=2,
                             min_delta=1e-4,
                             cooldown=1)

callbacks = [save_check, lr_decay]
print('start training')
vo.model.fit_generator(pose_generator(conf['data_path'], batch_size=10),
                       epochs=100,
                       steps_per_epoch=100,
                       verbose=1,
                       callbacks=callbacks)
