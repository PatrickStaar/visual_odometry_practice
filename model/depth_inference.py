from models import DepthModel
from util import *
from keras.callbacks import *
from conf import conf
from keras.models import model_from_json
from data_generator import depth_model_generator

# load json and create model

depth = DepthModel(input_shape=conf['input_shape'])
depth.model_from_file('depth_model_multi_output.json')

# Visualize
# depth.model.summary()
# plot_model(depth.model, to_file='depth_model.png')

depth.load_weights(conf['depth_weights'], resnet_only=True)
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

save_check = ModelCheckpoint(filepath='./weights/best.h5',
                             monitor='loss',
                             save_best_only=True,
                             save_weights_only=True,
                             mode='auto')
lr_decay = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2)

callbacks = [save_check, lr_decay]
print('start training')
depth.model.fit_generator(depth_model_generator(conf['data_path'], batch_size=2),
                          epochs=20,
                          steps_per_epoch=100,
                          verbose=1,
                          callbacks=callbacks)

