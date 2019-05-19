from depthmap import Depth
from conf import conf

with open(conf['data_path'][0] + 'depth.txt') as img_list:
    for line in img_list:
        line = line.rstrip('\n')
        timestamp, depth = line.split(' ')
        # timestamp = float(timestamp)
        d = Depth(conf['data_path'][0] + depth)
        d.interpolate()
        d.write(conf['data_path'][0] + 'interpolated/' + depth[6:])
