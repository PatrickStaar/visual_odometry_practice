import numpy as np

class Point:
    def __init__(self, coordinate, dimension=2):
        self.dimension=dimension
        self.x=coordinate[0]
        self.y=coordinate[1]
        self.z=0
        self.w=0

    def to_array(self):
        if self.dimension == 2:
            return np.array([self.x,self.y])
        return np.array([self.x,self.y,self.z])

    def transit(self, T):
        pass