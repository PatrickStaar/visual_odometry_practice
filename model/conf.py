conf = {
    'scale': 5000,
    'scale_factor': 1.2,
    'input_shape': (480, 640, 3),
    'batch_size': 4,
    'data_path': ['../../../datasets/freiburg1xyz/',
                  '../../../datasets/freiburg2xyz/',
                  '../../../datasets/freiburg3/',
                  # './datasets/nyu/'
                  ],
    'test_path': '../../../datasets/test/',
    'depth_model_for_pred': './json/depth_model_single_output.json',
    'depth_model_for_train': './json/depth_model_multi_output.json',
    'posenet': './json/posenet.json',

    'depth_weights': './weights/depth.h5',
    'pose_weights': './weights/pose.h5',
    'resnet': './weights/resnet50_notop.h5',
    'vgg': './weights/vgg.h5',
    'intrinsics': [525.0, 525.0, 319.5, 239.5, ],  # fx fy cx cy

}
