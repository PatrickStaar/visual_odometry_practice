from keras.layers import *
from keras import Model
from keras.applications.vgg16 import VGG16
from util import *
from keras.utils.vis_utils import plot_model
#import os
#os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin'

class OdometryModel():
    def __init__(self, inputs=None, input_shape=None):

        if input_shape is not None:
            self.input1=Input(input_shape)
            self.input2=Input(input_shape)
        else:
            self.input1=inputs[0]
            self.input2=inputs[1]

        self.input = concatenate([self.input1, self.input2])
        self.model=None

        self._build()

    def _build(self):

        vgg = VGG16(include_top=False, weights=None, input_tensor=self.input)

        x = vgg.output
        x = Flatten()(x)

        trans = Dense(units=4096, activation='relu', name='trans2')(x)
        trans = Dense(units=4096, activation='relu', name='trans1')(trans)
        trans = Dense(3, name='trans0')(trans)

        ori = Dense(units=4096, activation='relu', name='orient2')(x)
        ori = Dense(units=4096, activation='relu', name='orient1')(ori)
        ori = Dense(3,name='orient0')(ori)

        self.model=Model(inputs=[self.input1, self.input2], outputs=[trans, ori])



# vo=odometry_net(input_shape=(352,1216,3))
#vo.summary()
#plot_model(vo, to_file='./vo_model.png')

