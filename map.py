import numpy as np
from frame import Frame
from mappoint import Map_point


class Map:
    def __init__(self):
        self.pt_pool = {}
        self.frame_pool = {}

    def insert_frame(self, frame):
        self.frame_pool[frame.id] = frame

    def insert_pt(self, mappoint):
        self.pt_pool[mappoint.id] = mappoint

    def volume(self):
        return len(self.pt_pool)

    def get_desp(self):
        desp=[]
        for i in range(self.volume()):
            desp.append(self.pt_pool[i].destriptor)
        return np.array(desp)

    def pt3d_by_index(self, indexes):
        return np.array([self.pt_pool[i].position for i in indexes ])

