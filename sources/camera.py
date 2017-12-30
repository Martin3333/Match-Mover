# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np


R_ZERO = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])

class Camera(object):

    def __init__(self, K, R, T):
        self.K = K
        self.R = R
        self.T = T

    def P_from_RT(self):
        # print(self.K)
        # print(self.R)
        # print(self.T)
        return cv2.sfm.projectionFromKRt(self.K, self.R, self.T)

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
                if num_frames % 10 == 0:
                    image = str(frame_index) + ".jpg"
                    image_frame = os.path.join("..", "resources", "video_frames", image)
                    cv2.imwrite(image_frame, color)
                    frame_index += 1

                num_frames += 1
            else:
                break

        vcap.release()
        return frame_index

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
        F = cv2.sfm.normalizeFundamental(F)
        # Compute E.
        E = np.dot(np.dot(np.transpose(K), F), K)
        # Get R and T.
        _, R, T, mask = cv2.recoverPose(E, pts1, pts2, K)

        return R, T

    @staticmethod
    def find_frames_with_overlap(trajectories, start_frame, min_keypoints=100):

        kp_indices = [kp for kp in trajectories if start_frame in trajectories[kp]]

        overlap = kp_indices
        ctr = 0
        while len(overlap) > min_keypoints:
            ctr += 1
            overlap = [(kp,trajectories[kp]) for kp in kp_indices if start_frame+ctr in trajectories[kp]]
        ctr -= 1
        overlap = [(kp,trajectories[kp]) for kp in kp_indices if start_frame+ctr in trajectories[kp]]

        pts1 = []
        pts2 = []
        indices = []
        for o in overlap:
            pts1.append(o[1][start_frame])
            pts2.append(o[1][start_frame+ctr])
            indices.append(o[0])
        return ctr, pts1, pts2, indices

        

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
        matches = matcher.match(des1, des2)

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
    def find_cameras(trajectories, K, start_frame=1, camera_z_offset=+2000.0):

        cameras = {start_frame: Camera(K, R_ZERO, (0.0, 0.0, camera_z_offset))}

        offset1, pts1, pts2, indices1 = Camera.find_frames_with_overlap(trajectories, start_frame, min_keypoints=200)

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        R,T = Camera.get_camera_pose(pts1, pts2, K)
        T[2] += camera_z_offset

        cameras[start_frame+offset1] = Camera(K, R, T)

        while True:
            offset2, _, pts3, indices2 = Camera.find_frames_with_overlap(trajectories, start_frame, min_keypoints=100)
            if start_frame+offset2 in cameras:
                break

            pts1 = [trajectories[i][start_frame] for i in indices2]
            pts2 = [trajectories[i][start_frame+offset1] for i in indices2]

            P1 = cameras[start_frame].P_from_RT()
            P2 = cameras[start_frame+offset1].P_from_RT()
            F = cv2.sfm.fundamentalFromProjections(P1, P2)
            F = cv2.sfm.normalizeFundamental(F)
            
            pts1, pts2 = cv2.correctMatches(F, np.array([pts1]), np.array([pts2]))

            object_points = []
            for p1,p2 in list(zip(pts1[0], pts2[0])):
                ret = cv2.triangulatePoints(P1, P2, np.array([p1[0],p1[1]]), np.array([p2[0],p2[1]]))
                object_points.append(ret)
            object_points = cv2.convertPointsFromHomogeneous(np.array(object_points))
            ret, R, T, _ = cv2.solvePnPRansac(object_points, np.array(pts3), K, (0,0,0,0))
            R,_ = cv2.Rodrigues(R)
            cameras[start_frame+offset2] = Camera(K,R,T)

            for frame in range(start_frame, start_frame+offset2):
                if frame not in cameras:
                    pts_frame = [trajectories[kp][frame] for kp in indices2]
                    ret, R, T, _ = cv2.solvePnPRansac(object_points, np.array(pts_frame), K, (0,0,0,0))
                    cameras[frame] = Camera(K,R,T)

            start_frame += offset1
            offset1 = offset2 - offset1

        return cameras





    @staticmethod
    def detect_trajectories(video_file):

        trajectories = {}
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

            #Initially, every KP is the start of a trajectory
            for index in current_matches.keys():
                trajectories[index] = {0: old_keypoints[index].pt}



            image_number = 1

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
                        ## Already have that keypoint, add point for current frame
                        trajectories[keypoint_no][image_number] = new_keypoints[current_index].pt


                    else:
                        keypoint_no = keypoint_counter
                        keypoint_counter += 1
                        current_index = match.queryIdx
                        next_matches[current_index] = keypoint_no
                        ## New keypoint, add points from current and LAST frame, was there already but didn't recognize it
                        trajectories[keypoint_no] = {image_number: new_keypoints[current_index].pt,
                                                    image_number-1: old_keypoints[match.trainIdx].pt}


                current_matches = next_matches
                old_keypoints, old_descriptors = new_keypoints, new_descriptors



                image_number += 1
            else:
                break

        vcap.release()

        return trajectories
