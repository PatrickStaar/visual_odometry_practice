from keras import Model
from keras.layers import *
from util import *
from keras.applications import ResNet50, VGG16
from projective import vec2mat
from keras.models import model_from_json


def model_from_file(model_file):
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    return model_from_json(loaded_model_json)


def save_model(filename, model):
    model_json = model.to_json()
    with open(filename, "w") as json_file:
        json_file.write(model_json)


# nn for depth estimation
class DepthModel:
    def __init__(self, input_shape=None, input_tensor=None, mode='multi'):
        self.inputs = None
        self.model = None
        self.outputs = None
        self.mode = mode
        self.input_tensor = input_tensor
        self.input_shape = input_shape
        # self.build()

    def build(self):
        if self.input_shape is not None:
            input = Input(shape=self.input_shape)
        else:
            input = self.input_tensor

        x = ResNet50(include_top=False, weights=None, input_tensor=input)

        skip1 = x.get_layer('activation_1').output
        skip2 = x.get_layer('activation_10').output
        skip3 = x.get_layer('activation_22').output
        skip4 = x.get_layer('activation_40').output

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


        dec0 = activiate(dec0)
        if self.mode == 'multi':
            outputs = [dec0, dec1, dec2, dec3]
        else:
            outputs = [dec0]
        self.model = Model(inputs=[input], outputs=outputs)
        print('network built')

    def model_from_file(self, model_file):
        self.model = model_from_file(model_file)

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

        # vgg = VGG16(include_top=False, weights=None, input_tensor=self.input)
        # l = vgg.get_layer(name='block1_conv1')
        # l.name = 'block1_conv0'

        x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', kernel_initializer='uniform')(
            self.input)
        x = Conv2D(64, (5, 5), strides=(2, 2), padding='same', activation='relu', kernel_initializer='uniform')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(128, (3, 2), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x)
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x)
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x)
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x)
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x)
        x = MaxPooling2D((3, 3))(x)
        x = Flatten()(x)

        t = Dense(units=512, name='trans2')(x)
        t = Activation('tanh')(t)
        t = Dense(units=512, name='trans1')(t)
        t = Activation('tanh')(t)
        t = Dense(3, name='trans0')(t)
        # t = Activation('tanh')(t)

        r = Dense(units=512, name='orient2')(x)
        r = Activation('tanh')(r)
        r = Dense(units=512, name='orient1')(r)
        r = Activation('tanh')(r)
        r = Dense(3, name='orient0')(r)
        # r = Activation('tanh')(r)

        pose = Concatenate()([t, r])

        # if self.mode is not 'unsupervised':
        #     mat = Lambda(post_process)(mat)

        self.model = Model(inputs=[self.input1, self.input2], outputs=[pose])

    def load_weights(self, path, vgg_only=False):

        if vgg_only:
            self.model.load_weights(path, by_name=True)
        else:
            self.model.load_weights(path)

    def model_from_file(self, file):
        self.model = model_from_file(file)
