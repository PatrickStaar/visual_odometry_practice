from depth_model import DepthModel
from util import data_generator, paths_generator
from keras.losses import *
from keras.callbacks import *
from keras.utils.vis_utils import plot_model

train_data=r'D:\dev\kittit_kit\data_depth_selection\depth_selection\val_selection_cropped\images'

depth=DepthModel(input_shape=(352,1216,3))
depth.model.summary()
# depth.load_weights('./weights/resnet50_notop.h5',resnet_only=True)
# depth.model.compile(optimizer='adam', loss=mean_squared_error)
# plot_model(depth.model, to_file='depth_model.png')

## 一次性导入数据
# depth.model.fit()
# x, y=dataloader(base=r'D:\dev\kittit_kit\data_depth_selection\depth_selection\val_selection_cropped',
#                 train='image',
#                 groundtruth='groundtruth_depth')

## 使用生成器
# x=paths_generator(train_data)
#
# save_check = ModelCheckpoint(filepath='./weights/best.h5',
#                                   monitor='loss',
#                                   save_best_only=True,
#                                   save_weights_only=True,
#                                   mode='auto')
# lr_decay=ReduceLROnPlateau(monitor='loss',
#                            factor=0.1,
#                            patience=5)
#
# callbacks=[save_check,lr_decay]
#
#
# depth.model.fit_generator(data_generator(x,batch_size=2),
#                           epochs=20,
#                           steps_per_epoch=100,
#                           verbose=1,
#                           callbacks=callbacks)