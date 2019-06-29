import keras.backend as K
from models import *
from keras import Model
from keras.losses import mean_absolute_error, mean_squared_error
from projective import *
from conf import conf
from keras.optimizers import Adam


class VO:
    def __init__(self, input_shape, mode='train'):
        self.model = None
        self.input_shape = input_shape
        self.depth_shape = input_shape[:3]
        self.frame0 = None  # 当前帧
        self.frame1 = None  # 前一帧
        self.intrinsic = None
        self.mode = mode

    def build(self, frozen=False, separated_weights=False):
        self.intrinsic = get_intrinsics(conf['intrinsics'])
        frame0 = Input(batch_shape=self.input_shape, name='img0')
        frame1 = Input(batch_shape=self.input_shape, name='img1')

        dep = DepthModel(input_tensor=frame1, mode='single')
        dep.build()
        if frozen:
            for l in dep.model.layers:
                l.trainable = False

        odo = OdometryModel(inputs=[frame0, frame1])
        odo.build()

        if separated_weights:            
            dep.load_weights(conf['depth_weights'])
            odo.load_weights(conf['pose_weights'])
            print('weights_loaded')

        depthmap = dep.model.output
        pose = odo.model.output

        mat = Lambda(vec2mat, name='euler2mat')(pose)
        img_syn = Lambda(self.img_syn, name='synthesis'
                         )([frame1, depthmap, mat])


        inputs = [frame0, frame1]

        if self.mode is 'train':
            syn_loss = Lambda(self.syn_loss, name='syn_loss'
                              )([frame0, img_syn])

            # smo_loss = Lambda(self.smo_loss,
            #               name='smo_loss')(depthmap)

            outputs = [depthmap, pose, syn_loss]  # ,smo_loss]
        else:
            outputs = [depthmap, pose]

        self.model = Model(inputs=inputs,
                           outputs=outputs)


    def compile(self):
        syn_loss = self.model.get_layer('syn_loss').output
        # smo_loss = self.model.get_layer('smo_loss').output
        self.model.add_loss(syn_loss)
        #self.model.add_loss(smo_loss)
        adam = Adam(lr=0.01)
        self.model.compile(optimizer=adam,
                           loss=[None] * len(self.model.outputs), ) 

    def load_weights(self, paths):
        for file in paths:
            self.model.load_weights(file, by_name=True)

    def img_syn(self, inputs):
        img_src, depth, pose = inputs
        depth = K.clip(depth, 1e-6, 1e6)
        depth = 1.0 / depth
        img_tgt = synthesis(img_src, depth, pose, self.input_shape, self.intrinsic)
        return img_tgt

    def smo_loss(self, depthmap):
        loss_smo = smoothness(depthmap)
        return loss_smo

    def syn_loss(self, inputs):

        img_tgt, img_syn = inputs
        img_tgt_cropped = K.slice(img_tgt, (0, 40, 40, 0), (-1, 400, 560, -1))
        img_syn_cropped = K.slice(img_syn, (0, 40, 40, 0), (-1, 400, 560, -1))
        loss = K.mean(mean_absolute_error(img_tgt_cropped, img_syn_cropped))
        return loss


    def _expand(self, x, shape):
        x = K.expand_dims(x, axis=1)
        ones = K.zeros_like(x)
        # shape=K.cast(shape,'float32')
        times = K.cast(shape[1] * shape[2] / 16 - 1, 'float32')
        ones = K.tile(ones, [1, times, 1, 1])

        expanded = K.concatenate([x, ones], axis=1)
        expanded = K.reshape(expanded, shape)
        expanded = K.expand_dims(expanded)

        return expanded

    def model_from_file(self, model_file):
        self.model = model_from_file(model_file)

    def save_as_json(self, path):
        save_model(path, self.model)
