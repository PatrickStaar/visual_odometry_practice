from depth_model import DepthModel
from util import *
from keras.callbacks import *
from conf import conf

depth = DepthModel(input_shape=conf['input_shape'])

# Visualize
# depth.model.summary()
# plot_model(depth.model, to_file='depth_model.png')

depth.load_weights(conf['resnet_weights'], resnet_only=True)
print('weights loaded')
# depth.model.compile(optimizer='adam', loss=mean_squared_error)

losses = {'activation_50': 'mean_squared_error',
          'conv2d_transpose_4': 'mean_squared_error',
          'conv2d_transpose_3': 'mean_squared_error',
          'conv2d_transpose_2': 'mean_squared_error'}

loss_weights = {'activation_50': 1.0,
                'conv2d_transpose_4': 0.1,
                'conv2d_transpose_3': 0.05,
                'conv2d_transpose_2': 0.01}

depth.model.compile(optimizer='adam',
                    loss=losses,
                    loss_weights=loss_weights)
print('compiled')

## 使用生成器
# x=paths_generator(train_data)

# 一次性导入数据
# x, gt4,gt3,gt2,gt1 = data_tum(conf['data_path'],multi_losses=True)
# print('dataset loaded')

save_check = ModelCheckpoint(filepath='./weights/best.h5',
                             monitor='val_loss',
                             save_best_only=True,
                             save_weights_only=True,
                             mode='auto')
lr_decay = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2)

callbacks = [save_check, lr_decay]
print('start training')
depth.model.fit_generator(data_generator(conf['data_path'], batch_size=2),
                          epochs=20,
                          steps_per_epoch=100,
                          verbose=1,
                          callbacks=callbacks)

# depth.model.fit(x=x,
#                 y={'activation_50':gt4,
#                    'conv2d_transpose_4':gt3,
#                    'conv2d_transpose_3':gt2,
#                    'conv2d_transpose_2':gt1},
#                 batch_size=4,
#                 epochs=1,
#                 callbacks=callbacks,
#                 validation_split=0.05,
#                 )
