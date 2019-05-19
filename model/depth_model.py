from keras import Model
from keras.layers import *
from util import *
from keras.applications.resnet50 import ResNet50


class DepthModel:
    def __init__(self, input_shape=None, input_tensor=None, weights_path='./weights/dep_weights_best.h5'):
        self.inputs = None
        self.model = None
        self.nn = None
        self.callbacks = None
        self.params = None
        self.outputs = None
        self.input_tensor = input_tensor
        self.input_shape = input_shape
        self.weights_path = weights_path

        self._build()

    def _build(self):
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

        self.model = Model(inputs=[input], outputs=[dec0])
        print('network built')

    def model_from_file(self, model_file, weights_file=None):
        pass

    def predict(self, input):
        self.model.predict(input)

    def load_weights(self, path, resnet_only=False):

        if resnet_only:
            self.model.load_weights(path, by_name=True)
        else:
            self.model.load_weights(path)
