import keras.backend as K
from depth_model import DepthModel
from odometry import OdometryModel
from keras import Model
from keras.utils.vis_utils import plot_model
from keras.losses import mean_absolute_error
from projective import *


class VO:
    def __init__(self, input_shape, mode='train'):
        self.model = None
        self.input_shape = input_shape
        self.depth_shape = input_shape[:3]
        self.frame0 = None  # 当前帧
        self.frame1 = None  # 前一帧
        self.intrinsic = None
        self.mode = mode
        self._build()

    def _build(self):
        self.intrinsic = K.ones((1, 3, 3))
        frame0 = Input(batch_shape=self.input_shape)
        frame1 = Input(batch_shape=self.input_shape)

        dep = DepthModel(input_tensor=frame0)
        odo = OdometryModel(inputs=[frame0, frame1])

        depthmap = dep.model.output
        pose = odo.model.output

        # extended=Lambda(self._expand,arguments={'shape':self.depth_shape})(mat)
        if self.mode is 'train':

            # source=Concatenate()([self.frame0,depthmap,extended])
            loss_prj = Lambda(self.projective_loss,
                              # arguments={'intrinsic':self.intrinsic},
                              name='image_synthesis_loss'
                              )([frame0, frame1, depthmap, pose])

            inputs = [frame0, frame1]
            outputs = [depthmap, pose, loss_prj]
        else:
            inputs = [frame0, frame1]
            outputs = [depthmap, pose]

        self.model = Model(inputs=inputs,
                           outputs=outputs)

    def compile(self):
        projective_loss = self.model.get_layer('image_synthesis_loss').output
        self.model.add_loss(projective_loss)

        self.model.compile(optimizer='sgd',
                           loss=[None] * len(self.model.outputs))

    def projective_loss(self, inputs, intrinsic=None):
        """Inverse warp a source image to the target image plane based on projection.
        Args:
          source: the source image [batch, height_s, width_s, 3]
          depth: depth map of the target image [batch, height_t, width_t]
          pose: target to source camera transformation matrix [batch, 6], in the
                order of tx, ty, tz, rx, ry, rz
          intrinsics: camera intrinsics [batch, 3, 3]
        Returns:
          Source image inverse warped to the target image plane [batch, height_t,
          width_t, 3]
        """

        img_src, img_tgt, depth, pose = inputs
        batch, height, width, _ = self.input_shape

        def synthesis(img0, depth, pose, intrinsic):
            depth = K.squeeze(depth, axis=-1)
            intrinsics = K.tile(intrinsic, (batch, 1, 1))
            pixel_coords = meshgrid(batch, height, width)
            # 将像素坐标系转换为相机坐标系
            cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
            # 构造4x4内参矩阵
            filler = K.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
            filler = K.tile(filler, [batch, 1, 1])
            intrinsics = K.concatenate([intrinsics, K.zeros([batch, 3, 1])], axis=2)
            intrinsics = K.concatenate([intrinsics, filler], axis=1)
            # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
            # pixel frame.
            # 获得从目标帧到源帧的转换矩阵
            proj_tgt_cam_to_src_pixel = K.batch_dot(intrinsics, pose)
            src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
            # 获得合成后的图像
            img_syn = bilinear_sampler(img0, src_pixel_coords)
            return img_syn

        # img=source[:,:,:,:3]
        #
        # pose=K.reshape(source[:,:,:,4:],(batch,-1))
        #
        # pose=pose[:,:16]
        # pose=K.reshape(pose,(batch,4,4))

        # 建立像素坐标网格

        img_syn = synthesis(img_src, depth, pose, self.intrinsic)
        img_syn = K.reshape(img_syn, (batch, -1))
        img_tgt = K.reshape(img_tgt, (batch, -1))

        loss_syn = mean_absolute_error(img_tgt, img_syn)

        return loss_syn

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

# v = VO(input_shape=(1, 352, 1216, 3))
# plot_model(v.model, to_file='./vo.png', show_shapes=True)
# v.model.summary()
# v.compile()
