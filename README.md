# visual_odometry_demo
## This is a visual odometry project with deep-learning stuff

### Dependencies
* Keras
* Tensorflow
* Pillow
* Numpy

### Main Classes
1. DepthModel
> The input size of DepthModel is not fixed.
The default setting is [W:640,H:480]. 
Use `depth_inference.py` to train this model separately.
Depth ground truth should be contained in the dataset.
Dataset dirs should be like:
>
>./datasets
>
>--/item1
>
>----/rgb
>
>----/depth
>
>----/associated.txt
>
>In the `associated.txt`, the lines should be like`timestamp rgb/filename.jpg depth/filename.jpg`


 
2. OdometryModel
3. VO

### Other Files
Processed datasets and model related files are at `https://pan.baidu.com/s/1NLf8PgDxgpJmXeUrmDCr1A`, and code is `beq6`
