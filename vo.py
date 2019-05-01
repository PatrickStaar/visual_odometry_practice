import cv2
import numpy as np
from map import Map
from camera import Camera
from mappoint import Map_point
from matplotlib import pyplot as plt
import time


class vo:
    def __init__(self, config):
        self.camera = Camera(config)
        self.status = 0  # 0: uninitialized 1: initialized -1: lost
        self.cur_frame = None
        self.ref_frame = None
        self.cur_pts2d = None
        self.ref_pts2d = None
        self.ref_desp = None
        self.ref_pts3d = None
        self.map_matches = None
        self.R = None
        self.t = None
        self.matches = None
        self.detector = cv2.xfeatures2d.SURF_create(1000)
        self.T_wld2cam = np.eye(4)
        self.map = Map()

        self.set_matcher()

    def add_frame(self, frame, visual=False):

        self.cur_frame = frame
        self.cur_pts2d, self.cur_desp = self.detector.detectAndCompute(self.cur_frame.img, mask=None)

        if self.status == 0:
            self.set_ref()
            self.status = 1

        elif self.status == 1:
            self.get_matches()
            if visual and self.matches is not None:
                self.draw_matches()

            self.R, self.rvec, self.t = self.get_pose()

            self.T_cam = np.column_stack((self.R, np.reshape(self.t, (3, 1))))
            self.T_cam = np.row_stack((self.T_cam, np.array([0, 0, 0, 1], dtype=np.float32)))
            self.T_wld2cam = np.matmul(self.T_wld2cam, self.T_cam)
            self.set_ref()

        else:
            print('failed to track')

    def set_matcher(self):
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def set_ref(self):
        ref_pts3d = []
        ref_desp = []
        ref_pts2d = []
        for i in range(len(self.cur_pts2d)):
            d = self.cur_frame.get_depth(self.cur_pts2d[i])
            # print(d)
            if d > 0:
                ref_pts3d.append(self.camera.pix2cam(self.cur_pts2d[i], d))
                ref_desp.append(self.cur_desp[i])
                ref_pts2d.append(self.cur_pts2d[i])

        self.ref_pts3d = np.array(ref_pts3d, dtype=np.float32, )
        self.ref_desp = np.array(ref_desp, dtype=np.float32)
        self.ref_pts2d = ref_pts2d

        self.ref_frame = self.cur_frame



    def get_matches(self):
        matches = self.matcher.knnMatch(self.ref_desp, self.cur_desp, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.25 * n.distance:
                good_matches.append(m)
        self.matches = good_matches

    def draw_matches(self):
        m = cv2.drawMatches(self.ref_frame.img, self.ref_pts2d, self.cur_frame.img, self.cur_pts2d, self.matches[:20],
                            None)
        cv2.imwrite('./matches/match_{}.jpg'.format(self.ref_frame.id), m)

    def get_pose(self):
        matched_ref3d = []
        matched_cur2d = []
        for m in self.matches:
            matched_cur2d.append(np.array([self.cur_pts2d[m.trainIdx].pt[0], self.cur_pts2d[m.trainIdx].pt[1]]))
            matched_ref3d.append(self.ref_pts3d[m.queryIdx])
        succeed, rvec, t, x = cv2.solvePnPRansac(np.asanyarray(matched_ref3d), np.asanyarray(matched_cur2d),
                                                 self.camera.K, None)
        # print(vec)
        R = cv2.Rodrigues(rvec)[0]
        return R, rvec, t

     ####################################################
    ############## 使用局部地图 ###########################
   #####################################################

    def add2map(self):
        # 把3d点和描述子和像素坐标保存到map
        if self.map.volume() is 0:
            for i in range(len(self.cur_pts2d)):
                d = self.cur_frame.get_depth(self.cur_pts2d[i])
                # print(d)
                if d > 0:
                    mp = Map_point(id=self.map.volume(),
                                   coordinate=self.camera.pix2cam(self.cur_pts2d[i], d),
                                   descriptor=self.cur_desp[i],
                                   pixel=self.cur_pts2d[i])
                    self.map.insert_pt(mp)

        elif self.map.volume() > 1000:
            return

        else:
            # 把新帧中的特征点加入其中
            pass


    def map_match(self):
        ref_desp = self.map.get_desp()
        matches = self.matcher.knnMatch(ref_desp, self.cur_desp, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.25 * n.distance:
                good_matches.append(m)
        self.map_matches = good_matches

    def pose_from_map(self):
        ref_index = [i.queryIdx for i in self.map_matches]
        matched_ref3d = self.map.pt3d_by_index(ref_index)
        matched_cur2d = np.array([i.trainIdx for i in self.map_matches])

        succeed, rvec, t, x = cv2.solvePnPRansac(matched_ref3d, matched_cur2d, self.camera.K, None)
        # print(vec)
        R = cv2.Rodrigues(rvec)[0]
        return R, rvec, t

    def spin(self, frame):
        self.cur_frame = frame
        self.cur_pts2d, self.cur_desp = self.detector.detectAndCompute(self.cur_frame.img, mask=None)

        if self.status == 0:
            self.add2map()
            self.status = 1
        elif self.status == 1:
            self.map_match()

            self.R, self.rvec, self.t = self.get_pose()

            self.T_cam = np.column_stack((self.R, np.reshape(self.t, (3, 1))))
            self.T_cam = np.row_stack((self.T_cam, np.array([0, 0, 0, 1], dtype=np.float32)))
            self.T_wld2cam = np.matmul(self.T_wld2cam, self.T_cam)

            # 将已匹配的点的三维坐标更新，observed+1 corrected+1
            ## 需要引入新的深度，计算从pixel->camera->world的坐标

            # 计算map中未匹配点是否在视野中，不在则observed-1 否则+1
            ## 需要计算world->camera->pixel的坐标

            # 新帧中未匹配的点，配合深度值，描述子，替换掉map中不好的点

        else:
            print('failed to track')
