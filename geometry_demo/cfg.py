import json

class Config:
    def __init__(self, path):
        with open(path, 'r') as f:
            dic = json.load(f)
            self.fx = dic['fx']
            self.fy = dic['fy']
            self.cx = dic['cx']
            self.cy = dic['cy']
            self.path = dic['data_path']
            self.depth_scale = dic['scale']

    def create_file(self, path):
        pass

