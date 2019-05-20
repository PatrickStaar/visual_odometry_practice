conf = {
    'fx': 517.3,
    'fy': 516.5,
    'cx': 325.1,
    'cy': 249.7,
    'scale': 5000,
    'scale_factor': 1.2,
    'input_shape': (480, 640, 3),
    'batch_size': 4,
    'data_path': ['./datasets/freiburg1xyz/',
                  './datasets/freiburg2xyz/',
                  './datasets/freiburg3/',
                  './datasets/nyu/'
                  ],
    'depth_weights': './weights/depth.h5',
    'depth_model_for_pred': './json/depth_model_single_output.json',
    'depth_model_for_train': './json/depth_model_multi_output.json',
    'pose_weights': './weights/pose.h5',
    'resnet': './weights/resnet50_notop.h5',
    'vgg': './weights/vgg.h5',

}
