from keras import Model
from keras.layers import *
from util import *


class DepthModel:
    def __init__(self, input_shape, weights_path='./weights/dep_weights_best.h5'):
        self.inputs = None
        self.model = None
        self.nn = None
        self.callbacks = None
        self.params = None
        self.outputs = None
        self.input_shape = input_shape
        self.weights_path = weights_path

        self._build()

    def _build(self):

        input = Input(shape=self.input_shape)
        res1 = ZeroPadding2D((3, 3))(input)
        res1 = conv(res1, 64, (7, 7), (2, 2), name='conv1')
        res1 = batchnorm(res1, name='bn_conv1', axis=3)
        res1 = activiate(res1)
        #res1 = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(res1)
        res1_pooling = pooling(res1, size=(2, 2))
        
        stage = 2
        filters = [64, 64, 256]
        res2 = res_conv_block(res1_pooling, filters, stage, block='a', stride=(1, 1))
        res2 = res_identity_block(res2, filters, stage, block='b')
        res2 = res_identity_block(res2, filters, stage, block='c')
    
        
        stage = 3
        filters = [128, 128, 512]
        res3 = res_conv_block(res2, filters, stage, block='a')
        res3 = res_identity_block(res3, filters, stage, block='b')
        res3 = res_identity_block(res3, filters, stage, block='c')
        res3 = res_identity_block(res3, filters, stage, block='d')

        stage = 4
        filters = [256, 256, 1024]
        res4 = res_conv_block(res3, filters, stage, block='a')
        res4 = res_identity_block(res4, filters, stage, block='b')
        res4 = res_identity_block(res4, filters, stage, block='c')
        res4 = res_identity_block(res4, filters, stage, block='d')
        res4 = res_identity_block(res4, filters, stage, block='e')
        res4 = res_identity_block(res4, filters, stage, block='f')

        stage = 5
        filters = [512, 512, 2048]
        res5 = res_conv_block(res4, filters, stage, block='a')
        res5 = res_identity_block(res5, filters, stage, block='b')
        res5 = res_identity_block(res5, filters, stage, block='c')

        # res=pooling(res,size=(2,2),type='avg',padding='same')

        code = conv(res5, 1, (1, 1), (1, 1), name='conv_post_res')

        skip1 = conv(res1, 1, (1, 1), (1, 1))
        skip2 = conv(res2, 1, (1, 1), (1, 1))
        skip3 = conv(res3, 1, (1, 1), (1, 1))
        skip4 = conv(res4, 1, (1, 1), (1, 1))
        # skip5 = conv(res5, 1, (1, 1), (1, 1))

        dec4 = decoder_block(code, skip4, [1, ], 4)
        dec3 = decoder_block(dec4, skip3, [1, ], 3)
        dec2 = decoder_block(dec3, skip2, [1, ], 2)
        dec1 = decoder_block(dec2, skip1, [1, ], 1)

        dec0 = decoder_block(dec1, None, [1, ], 0)
        dec0 = activiate(dec0)

        self.model = Model(inputs=input, outputs=dec0)
        print('network built')

    def model_from_file(self, model_file, weights_file=None):
        pass

    def predict(self, input):
        self.model.predict(input)

    def load_weights(self,path,resnet_only=False):

        if resnet_only:
            self.model.load_weights(path,by_name=True)
        else:
            self.model.load_weights(path)


    # def get_data(self, data):
    #     return data
