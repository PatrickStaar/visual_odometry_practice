import numpy as np
from point import Point
import cv2


class Camera:
    def __init__(self, config):
        self.fx = config.fx
        self.fy = config.fy
        self.cx = config.cx
        self.cy = config.cy
        self.depth_scale = config.depth_scale
        self.K = self.getK()

    def wld2cam(self, pt_wld, T):
        return pt_wld.transit(T)

    def cam2wld(self, pt_cam, T):
        return pt_cam.transit(T)

    def cam2pix(self, pt_cam):
        x = pt_cam.x * self.fx / pt_cam.z + self.cx
        y = pt_cam.y * self.fy / pt_cam.z + self.cy
        return Point([x, y])

    def pix2cam(self, kp, depth=1):
        x, y = self.move(kp)
        x = x * depth / self.fx
        y = y * depth / self.fy
        z = depth
        return [x, y, z]

    def pix2wld(self, pt, T, depth):
        return self.cam2wld(self.pix2cam(pt, depth), T)

    def wld2pix(self, pt_wld, T):
        return self.cam2pix(self.wld2cam(pt_wld, T))

    def getK(self):
        K = np.zeros((3, 3))
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[2, 2] = 1.0
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        return K

    def move(self, kp):
        x, y = kp.pt
        x = x - self.cx
        y = y - self.cy
        return np.array([x, y])
