from keras.layers import *
from keras import Model
from keras.applications.vgg16 import VGG16
from util import *
from projective import vec2mat
from keras.utils.vis_utils import plot_model


class OdometryModel():

    def __init__(self, inputs=None, input_shape=None, mode='unsupervised'):

        self.mode = mode
        if input_shape is not None:
            self.input1 = Input(batch_shape=input_shape)
            self.input2 = Input(batch_shape=input_shape)
        else:
            self.input1 = inputs[0]
            self.input2 = inputs[1]

        self.input = concatenate([self.input1, self.input2])
        self.model = None

        self._build()

    def _build(self):

        res = VGG16(include_top=False, weights=None, input_tensor=self.input)
        x = res.output
        x = AveragePooling2D((3, 3))(x)
        x = Flatten()(x)

        # trans = Dense(units=2048, activation='relu', name='trans2')(x)
        # trans = Dense(units=2048, activation='relu', name='trans1')(trans)
        # trans = Dense(3, name='trans0')(trans)
        #
        # ori = Dense(units=2048, activation='relu', name='orient2')(x)
        # ori = Dense(units=2048, activation='relu', name='orient1')(ori)
        # ori = Dense(3,name='orient0')(ori)
        #
        # pose=Concatenate()([trans,ori])

        trans = Dense(units=2048, activation='relu', name='trans2')(x)
        trans = Dense(units=1024, activation='relu', name='trans1')(trans)
        trans = Dense(6, name='trans0')(trans)
        mat = Lambda(vec2mat, name='euler2mat')(trans)

        if self.mode is not 'unsupervised':
            mat = Lambda(post_process)(mat)

        self.model = Model(inputs=[self.input1, self.input2], outputs=[mat])
