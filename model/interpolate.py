from depthmap import Depth
from conf import conf

base = conf['data_path'][-1]

# 处理 raw depth 图片
with open(base + 'depth.txt', 'r') as f:
    deps = f.readlines()
    for d in deps:
        stamp, x = d.split(' ')
        x = x.strip('\n')
        dep = Depth(filename=base + x)
        dep.interpolate()
        dep.write(base + x.replace('depth.', 'interp.'))

# from matplotlib import pyplot as p
# dep = Depth(filename=r'D:\Code\VO\VO\model\datasets\washington\depth\scene_01\00000-interp.png')

# x=dep.depthmap
# p.imsave('x.jpg',x)
# # dep.interpolate()
# y=dep.depthmap

# p.imshow(y)
# p.colorbar()
# p.show()
