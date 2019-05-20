from keras import Model
from keras.layers import *
from util import *
from keras.applications import ResNet50, VGG16
from projective import vec2mat
from keras.models import model_from_json


# nn for depth estimation
class DepthModel:
    def __init__(self, input_shape=None, input_tensor=None, mode='multi'):
        self.inputs = None
        self.model = None
        self.outputs = None
        self.mode = mode
        self.input_tensor = input_tensor
        self.input_shape = input_shape

    def build(self):
        if self.input_shape is not None:
            input = Input(shape=self.input_shape)
        else:
            input = Input(tensor=self.input_tensor)

        x = ResNet50(include_top=False, weights=None, input_tensor=input)
        # x.summary()
        skip1 = x.get_layer('activation_1').output
        skip2 = x.get_layer('activation_10').output
        skip3 = x.get_layer('activation_22').output
        skip4 = x.get_layer('activation_40').output
        # skip5 = x.ger_layer('activiation_98').output

        skip1 = conv(skip1, 1, (1, 1), (1, 1))
        skip2 = conv(skip2, 1, (1, 1), (1, 1))
        skip3 = conv(skip3, 1, (1, 1), (1, 1))
        skip4 = conv(skip4, 1, (1, 1), (1, 1))

        code = conv(x.get_layer('activation_49').output, 1, (1, 1), (1, 1), name='conv_post_res')

        dec4 = deconv(code, 1, kernel=(4, 4), stride=(2, 2))
        x = Add()([dec4, skip4])
        dec3 = deconv(x, 1, kernel=(4, 4), stride=(2, 2))
        x = Add()([dec3, skip3])
        dec2 = deconv(x, 1, kernel=(4, 4), stride=(2, 2))
        x = Add()([dec2, skip2])
        dec1 = deconv(x, 1, kernel=(4, 4), stride=(2, 2))
        x = Add()([dec1, skip1])
        dec0 = deconv(x, 1, kernel=(4, 4), stride=(2, 2))

        #
        # dec4 = decoder_block(code, skip4, [1, ], 4)
        # dec3 = decoder_block(dec4, skip3, [1, ], 3)
        # dec2 = decoder_block(dec3, skip2, [1, ], 2)
        # dec1 = decoder_block(dec2, skip1, [1, ], 1)
        #
        # dec0 = decoder_block(dec1, None, [1, ], 0)
        dec0 = activiate(dec0)
        if self.mode == 'multi':
            outputs = [dec0, dec1, dec2, dec3]
        else:
            outputs = [dec0]
        self.model = Model(inputs=[input], outputs=outputs)
        print('network built')

    def model_from_file(self, model_file):
        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)

    def predict(self, input):
        self.model.predict(input)

    def load_weights(self, path, resnet_only=False):

        if resnet_only:
            self.model.load_weights(path, by_name=True)
        else:
            self.model.load_weights(path)


# nn for pose estimation
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

        x = Conv2D(filters=64, kernel_size=(7, 7), padding='same', activation='relu')(self.input)

        res = VGG16(include_top=False, weights=None, input_tensor=self.input)
        l = res.get_layer(name='block1_conv1')
        l.name = 'block1_conv0'
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

        # if self.mode is not 'unsupervised':
        #     mat = Lambda(post_process)(mat)

        self.model = Model(inputs=[self.input1, self.input2], outputs=[mat])

    def load_weights(self, path, vgg_only=False):

        if vgg_only:
            self.model.load_weights(path, by_name=True)
        else:
            self.model.load_weights(path)
