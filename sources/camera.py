# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np


class Camera(object):
    @staticmethod
    def generate_video_frames(video_file):
        frame_index = 0
        num_frames = 0

        vcap = cv2.VideoCapture(video_file)

        if vcap is None:
            print("The video capture property could not be instantiated.")
            quit()

        while vcap.isOpened():
            ret, frame = vcap.read()

            if ret:
                color = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                if num_frames % 20 == 0:
                    image = str(frame_index) + ".jpg"
                    image_frame = os.path.join("..", "resources", "video_frames", image)
                    cv2.imwrite(image_frame, color)
                    frame_index += 1

                num_frames += 1
            else:
                break

        vcap.release()

    @staticmethod
    def calibrate(max_index, chess_board_rows, chess_board_columns,
                  path=os.path.join("..", "resources", "video_frames")):
        # Termination criteria.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare the object points, like (0,0,0), (1,0,0), (2,0,0), ....,(6,5,0).
        obj_pt = np.zeros((chess_board_rows * chess_board_columns, 3), np.float32)
        obj_pt[:, :2] = np.mgrid[0:chess_board_columns, 0:chess_board_rows].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        obj_points = []  # 3d point in real world space.
        img_points = []  # 2d points in image plane.

        for i in range(0, max_index):
            # Read the image.
            image = str(i) + ".jpg"
            image_frame = os.path.join(path, image)
            img = cv2.imread(image_frame, 0)

            # Find the chess board corners.
            ret, corners = cv2.findChessboardCorners(img, (chess_board_rows, chess_board_columns), None)

            # If found, add object points, image points (after refining them).
            if ret:
                obj_points.append(obj_pt)
                cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
                img_points.append(corners)

        img = cv2.imread(os.path.join(path, "0.jpg"), 0)
        ret, mtx, dist, r_vectors, t_vectors = cv2.calibrateCamera(obj_points, img_points, img.shape[::-1], None, None)
        return mtx

    @staticmethod
    def get_camera_pose(pts1, pts2, K):
        # Compute F
        F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC)
        # Compute E
        E = np.dot(np.dot(np.transpose(K), F), K)
        # Get R and T
        _, R,T, mask= cv2.recoverPose(E, pts1, pts2, K)

        return R,T

    @staticmethod
    def get_keypoints(img):

        detector = cv2.xfeatures2d.SURF_create()
        kp, des = detector.detectAndCompute(img, None)

        # for k in kp:
        #     cv2.circle(img, (int(k.pt[0]),int(k.pt[1])),5, (255,255,255))

        # cv2.imshow('sub pixel', img)
        # if cv2.waitKey(0) & 0xff == 27:
        #     cv2.destroyAllWindows()

        return kp, des

    @staticmethod
    def match_keypoints(des1, des2):

        matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=True)
        matches = matcher.match(des2, des1)

        return matches



    @staticmethod
    def get_keypoints_harris(img):

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = np.float32(gray)

        dst = cv2.cornerHarris(gray, 3, 3, 0.1)
        dst = cv2.dilate(dst, None)
        #print(np.average(dst))
        ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)


        dst = np.uint8(dst)

        # Find the centroids.
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        # Define the criteria to stop and refine the corners.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

        # for c in corners:
        #     cv2.circle(img, (int(c[0]),int(c[1])),5, (255,255,255))

        # cv2.imshow('sub pixel', img)
        # if cv2.waitKey(0) & 0xff == 27:
        #     cv2.destroyAllWindows()

        return corners


