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
