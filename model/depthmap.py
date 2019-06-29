from PIL import Image
import numpy as np


class Depth:
    def __init__(self, filename=None):
        self.depthmap = None
        self.height = None
        self.width = None
        self.filename = filename

        self.read()

    def set_depth(self, row, col, value):
        self.depthmap[row, col] = value

    def get(self, row, col):
        return self.depthmap[row, col]

    def valid(self, row, col):
        if self.get(row, col) <= 0:
            return False
        else:
            return True

    # 补充深度图中的缺失部分
    def interpolate(self):
        height = self.height
        width = self.width

        for r in range(height):
            waitlist = 0
            for c in range(width):
                if self.valid(r, c):
                    if waitlist > 0:
                        void0 = c - waitlist
                        void1 = c - 1
                        if void0 > 0 and void1 < width - 1:
                            min_dep = min(self.get(r, void0 - 1), self.get(r, c))
                            for i in range(void0, c):
                                self.set_depth(r, i, min_dep)
                    waitlist = 0
                else:
                    waitlist += 1

            for c in range(width):
                if self.valid(r, c):
                    dep = self.get(r, c)
                    for i in range(0, c):
                        self.set_depth(r, i, dep)
                    break

            for c in range(width - 1, -1, -1):
                if self.valid(r, c):
                    dep = self.get(r, c)
                    for i in range(c + 1, width):
                        self.set_depth(r, i, dep)
                    break

        for c in range(width):
            for r in range(height):
                if self.valid(r, c):
                    dep = self.get(r, c)
                    for i in range(0, r):
                        self.set_depth(i, c, dep)
                    break

            for r in range(height - 1, -1, -1):
                if self.valid(r, c):
                    dep = self.get(r, c)
                    for i in range(r + 1, height):
                        self.set_depth(i, c, dep)
                    break

    def read(self):
        # loads depth map D from png file
        # and returns it as a numpy array,
        depth_png = np.array(Image.open(self.filename), dtype=int)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert (np.max(depth_png) > 255)

        depth = depth_png.astype(np.float) / 5000.
        depth[depth_png == 0] = -1.
        # depth=depth.transpose()
        self.height = depth.shape[0]
        self.width = depth.shape[1]
        self.depthmap = depth

    def write(self, filename='./interpolated_depth.png'):
        dep = self.depthmap * 5000
        dep = dep.astype(np.int)
        img = Image.fromarray(dep)
        img.save(filename)
