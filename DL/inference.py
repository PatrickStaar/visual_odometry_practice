from depth_model import DepthModel
from util import *
from keras.callbacks import *
from keras.losses import mean_squared_error
from conf import conf

depth = DepthModel(input_shape=conf['resnet_weights'])

# Visualize
# depth.model.summary()
# plot_model(depth.model, to_file='depth_model.png')

depth.load_weights(conf['resnet_weights'], resnet_only=True)
print('weights loaded')
depth.model.compile(optimizer='adam', loss=mean_squared_error)
print('compiled')

## 使用生成器
# x=paths_generator(train_data)

# 一次性导入数据
x, y = data_tum('./interpolated/')
print('dataset loaded')

save_check = ModelCheckpoint(filepath='./weights/best.h5',
                             monitor='loss',
                             save_best_only=True,
                             save_weights_only=True,
                             mode='auto')
lr_decay = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5)

callbacks = [save_check, lr_decay]

# depth.model.fit_generator(data_generator(x,batch_size=2),
#                           epochs=20,
#                           steps_per_epoch=100,
#                           verbose=1,
#                           callbacks=callbacks)
print('start training')
depth.model.fit(x=x,
                y=y,
                batch_size=4,
                epochs=1,
                callbacks=callbacks)
