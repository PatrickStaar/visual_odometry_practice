
class Frame:
    def __init__(self, id, camera,  img, depth,stamp=None, pose=None):
        self.id = id
        self.timestamp = stamp
        self.camera = camera
        self.T_w2c = pose
        self.img = img
        self.depth = depth

    def get_depth(self, kp):
        x, y = kp.pt
        x = round(x)
        y = round(y)
        d = self.depth[y, x]
        # print(d)
        if d > 0:
            return d / self.camera.depth_scale
        else:
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if i == 0 and j == 0:
                        continue

                    neighbor = [y + i, x + j]
                    if self.out_of_bounds(neighbor[0], neighbor[1]):
                        continue

                    dd = self.depth[neighbor[0], neighbor[1]]

                    if dd > 0:
                        # print(dd)
                        return dd / self.camera.depth_scale

        return -1

    def get_cam_center(self):
        # I don't know what this is for
        pass

    def in_frame(self, pt_wld):
        pt_cam = self.camera.wld2cam(pt_wld, self.T_w2c)
        if pt_cam.z <= 0:  # 该点在镜头之后
            return False
        pt = self.camera.cam2pix(pt_cam)
        return not self.out_of_bounds(pt.x, pt.y)

    def out_of_bounds(self, y, x):
        if x >= 0 and y >= 0 and x < self.img.shape[1] and y < self.img.shape[0]:
            return False
        else:
            return True
