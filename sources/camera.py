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

        if vcap is None or not vcap.isOpened():
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
        # Compute F.
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC)
        # Compute E.
        E = np.dot(np.dot(np.transpose(K), F), K)
        # Get R and T.
        _, R, T, mask = cv2.recoverPose(E, pts1, pts2, K)

        return R, T

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

    @staticmethod
    def pickle_keypoints(keypoints, descriptors):
        i = 0
        temp_array = []
        for point in keypoints:
            temp = (point.pt, point.size, point.angle, point.response, point.octave,
                    point.class_id, descriptors[i])
            i += 1
            temp_array.append(temp)
        return temp_array

    @staticmethod
    def unpickle_keypoints(array):
        keypoints = []
        descriptors = []
        for point in array:
            temp_feature = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2],
                                        _response=point[3], _octave=point[4], _class_id=point[5])
            temp_descriptor = point[6]
            keypoints.append(temp_feature)
            descriptors.append(temp_descriptor)
        return keypoints, np.array(descriptors)

    @staticmethod
    def unpickle_all_keypoints(array):
        keypoints = []
        descriptors = []
        for i in range(len(array)):
            kp, desc = Camera.unpickle_keypoints(array[i])
            keypoints.append(kp)
            descriptors.append(desc)
        return keypoints, descriptors

    @staticmethod
    def detect_keypoints(video_file):
        keypoints = []
        all_matches = {}
        vcap = cv2.VideoCapture(video_file)

        if vcap is None or not vcap.isOpened():
            print("The video capture property could not be instantiated.")
            quit()
        else:
            ret, old_frame = vcap.read()
            old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            old_keypoints, old_descriptors = Camera.get_keypoints(old_frame)

            keypoint_counter = len(old_keypoints)
            current_matches = dict((i, i) for i in range(len(old_keypoints)))

            keypoints.append(Camera.pickle_keypoints(old_keypoints, old_descriptors))

            image_number = 0

        while vcap.isOpened():
            ret, new_frame = vcap.read()

            if ret:
                new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
                new_keypoints, new_descriptors = Camera.get_keypoints(new_frame)

                if new_descriptors is not None and old_descriptors is not None:
                    matches = Camera.match_keypoints(new_descriptors, old_descriptors)
                else:
                    matches = []

                # Each match produces:
                # =========================================
                # queryIdx - index of keypoint in new_keypoints.
                # trainIdx - index of keypoint in old_keypoints.

                # Two possibilities for each match:
                # =========================================
                # (1) the match is new, which means, that we haven't tracked that keypoint before.
                # (2) the match is a continuation, which means, that we've tracked that keypoint before.

                next_matches = {}

                for match in matches:
                    if match.trainIdx in current_matches:
                        keypoint_no = current_matches[match.trainIdx]
                        current_index = match.queryIdx
                        next_matches[current_index] = keypoint_no
                    else:
                        keypoint_no = keypoint_counter
                        keypoint_counter += 1
                        current_index = match.queryIdx
                        next_matches[current_index] = keypoint_no

                current_matches = next_matches
                old_keypoints, old_descriptors = new_keypoints, new_descriptors

                keypoints.append(Camera.pickle_keypoints(old_keypoints, old_descriptors))


                #Comment these out to see that there are always less matched keypoints than detected ones
                # print('Matches: ' + str(len(current_matches)))
                # print('Keypoints: ' + str(len(new_keypoints)))

                # TODO fix IndexError: list index out of range in line 192.
                # for current_index, keypoint_no in current_matches.items():
                #    keypoint = new_keypoints[current_index]
                #    print(image_number, keypoint_no, keypoint.pt[0], keypoint.pt[1])

                image_number += 1
            else:
                break

        vcap.release()

        return keypoints
