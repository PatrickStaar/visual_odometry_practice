import numpy as np
import random

from keras.models import Model, Sequential
from keras.layers import *
from keras.callbacks import *
from keras.losses import *
from keras.optimizers import SGD

# Sequential
# model=Sequential()
# # model.add(Input(shape=(100,)))
# model.add(Dense(units=64,activation='relu',input_dim=1,name='den1'))
# model.add(Dense(units=1,activation='relu',name='dense2'))


# Functional
inputs = Input(shape=(1,))
x = Dense(units=64)(inputs)
x = Activation(activation='relu')(x)
x = Dense(units=1)(x)
outputs = Activation(activation='relu')(x)

model = Model(inputs=inputs,
              outputs=outputs)

# Compile
sgd = SGD(lr=0.01)
path = './weights_.hdf5'
checkpoint = ModelCheckpoint(filepath=path,
                             monitor='loss',
                             save_best_only=True,
                             save_weights_only=True,
                             mode='auto')

model.compile(optimizer=sgd,
              loss=mean_squared_error)

# preparing data
x = np.arange(0, 1, 0.00001)
print(x.shape)
y = []
for i in x:
    y.append(3 * i + random.random() / 100)
y = np.array(y)

# train
model.fit(x=x,
          y=y,
          batch_size=32,
          epochs=10,
          callbacks=[checkpoint],)

# predict
# model.load_weights('./weights_10.hdf5')
# print(model.predict(np.array([0.2,0.3])))
