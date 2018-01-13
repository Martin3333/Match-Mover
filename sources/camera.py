# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from root_sift import RootSIFT


R_ZERO = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


class Camera(object):
    def __init__(self, K, R, T):
        self.K = K
        self.R = R
        self.T = T

    def P_from_RT(self):

        #print(np.dot(self.K, np.dot(self.R, np.array(self.T).T)))
        #return np.dot(self.K, np.dot(self.R, self.T))
        return cv2.sfm.projectionFromKRt(self.K, self.R, self.T)
    @property
    def r_vec(self):
        return cv2.Rodrigues(self.R)[0]

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
    def calibrate(max_index, chess_board_rows, chess_board_columns,path=os.path.join("..", "resources", "video_frames")):
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
                corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
                img_points.append(corners)

        img = cv2.imread(os.path.join(path, "0.jpg"), 0)
        ret, mtx, dist, r_vectors, t_vectors = cv2.calibrateCamera(obj_points, img_points, img.shape[::-1], None, None)
        return mtx

    @staticmethod
    def get_camera_pose(pts1, pts2, K):
        # Compute F.
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, param1=3.5)
        #print(mask)
        F = cv2.sfm.normalizeFundamental(F)
        # Compute E.
        E = np.dot(np.dot(np.transpose(K), F), K)
        # Get R and T.
        _, R, T, _ = cv2.recoverPose(E, pts1, pts2, K)


        return R, T, mask

    @staticmethod
    def find_frames_with_overlap(trajectories, start_frame, min_keypoints=100):
        kp_indices = [kp for kp in trajectories if start_frame in trajectories[kp]]

        overlap = kp_indices
        ctr = 0
        while len(overlap) > min_keypoints:
            ctr += 1
            overlap = [(kp, trajectories[kp]) for kp in kp_indices if start_frame + ctr in trajectories[kp]]
        ctr -= 1
        overlap = [(kp, trajectories[kp]) for kp in kp_indices if start_frame + ctr in trajectories[kp]]

        pts1 = []
        pts2 = []
        indices = []
        for o in overlap:
            pts1.append(o[1][start_frame])
            pts2.append(o[1][start_frame + ctr])
            indices.append(o[0])

        return ctr, pts1, pts2, indices

    @staticmethod
    def get_keypoints(image):
        # detector = cv2.xfeatures2d.SURF_create()
        # kps, descs = detector.detectAndCompute(img, None)


        # Here are two different variants of feature detection and extraction.
        # To test them, please comment one of these variants out.

        # ================ Variant No. 1 ================ #

        # Detect difference of Gaussian keypoints in the image.
        detector = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create()
        kps = detector.detect(image)

        # Extract RootSIFT descriptors.
        rs = RootSIFT()
        kps, descs = rs.compute(image, kps)

        # =============================================== #

        # ================ Variant No. 2 ================ #

        # Initialize FAST feature detector.
        #fast = cv2.FastFeatureDetector_create(threshold=35)
        #kps = fast.detect(image, None)

        # Extract RootSift descriptors.
        #rs = RootSIFT()
        #kps, descs = rs.compute(image, kps)

        # =============================================== #

        return kps, descs

    @staticmethod
    def match_keypoints(des1, des2):
        matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=True)
        matches = matcher.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # Apply ratio test.
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])

        return good_matches



    @staticmethod
    def find_keyframe_cameras(trajectories, K, start_frame=1, camera_z_offset=2000.0):

        cameras = {start_frame: Camera(K, R_ZERO, (0.0, 0.0, camera_z_offset))}
        points_3d = {}

        offset1, pts1, pts2, indices1 = Camera.find_frames_with_overlap(trajectories, start_frame, min_keypoints=200)

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

        #add to points 3D
        R,T, mask = Camera.get_camera_pose(pts1, pts2, K)
        T[2] += camera_z_offset

        cameras[start_frame + offset1] = Camera(K, R, T)




        while True:
            offset2, _, pts3, indices2 = Camera.find_frames_with_overlap(trajectories, start_frame, min_keypoints=100)
            if start_frame + offset2 in cameras:
                break

            pts1 = [trajectories[i][start_frame] for i in indices2]
            pts2 = [trajectories[i][start_frame + offset1] for i in indices2]

            P1 = cameras[start_frame].P_from_RT()
            P2 = cameras[start_frame + offset1].P_from_RT()
            F = cv2.sfm.fundamentalFromProjections(P1, P2)
            F = cv2.sfm.normalizeFundamental(F)

            pts1, pts2 = cv2.correctMatches(F, np.array([pts1]), np.array([pts2]))

            object_points = []
            for p1, p2 in list(zip(pts1[0], pts2[0])):
                ret = cv2.triangulatePoints(P1, P2, np.array([p1[0], p1[1]]), np.array([p2[0], p2[1]]))
                object_points.append(ret)

            object_points = cv2.convertPointsFromHomogeneous(np.array(object_points))
            ret, R, T, inliers = cv2.solvePnPRansac(object_points, np.array(pts3), K, (0,0,0,0), reprojectionError=20.0)
            R,_ = cv2.Rodrigues(R)
            cameras[start_frame+offset2] = Camera(K,R,T)


            for i in range(0,len(indices2)-1):
                if indices2[i] not in points_3d and i in inliers:
                    points_3d[indices2[i]] = object_points[i]


            start_frame += offset1
            offset1 = offset2 - offset1

        return cameras, points_3d

    @staticmethod
    def find_all_cameras(trajectories, cameras, points_3d, K):

        for frame in range(0, max(cameras.keys())):
            if frame not in cameras:
                indices = [kp for kp in trajectories if frame in trajectories[kp] and kp in points_3d]
                pts_frame = [trajectories[kp][frame] for kp in indices]
                # if frame-1 in cameras:
                #     ret, R, T, _ = cv2.solvePnPRansac(np.array([points_3d[kp] for kp in indices]), np.array(pts_frame), K, (0,0,0,0), cameras[frame-1].r_vec, cameras[frame-1].T, useExtrinsicGuess=True)
                # else:
                ret, R, T, _ = cv2.solvePnPRansac(np.array([points_3d[kp] for kp in indices]), np.array(pts_frame), K, (0,0,0,0), reprojectionError = 10.0)
                R,_ = cv2.Rodrigues(R)
                cameras[frame] = Camera(K,R,T)
        return cameras


    #TODO: sparse and filter outliers
    @staticmethod
    def bundle_adjustment(cameras, trajectories, points_3d, K):
        filtered_trajectories = {}
        for p in trajectories:
            if p in points_3d:
                filtered_trajectories[p] = {}
                for frame in trajectories[p]:
                    if frame in cameras:
                        filtered_trajectories[p][frame] = trajectories[p][frame]
        #print(filtered_trajectories)

        n_cameras = len(cameras.keys())
        n_points = len(points_3d.keys())
        camera_params = []
        for c in cameras.values():
            R = cv2.Rodrigues(c.R)[0]
            camera_params.append([R[0][0], R[1][0], R[2][0], c.T[0], c.T[1], c.T[2]])

        camera_params = np.array(camera_params)
        point_indices = [p for p in points_3d.keys()]
        camera_indices = [c for c in cameras.keys()]
        points_3d_params = np.array([p[0] for p in points_3d.values()])

        def fun(params, n_cameras, n_points, point_indices, camera_indices, trajectories, K):
            camera_params = params[:n_cameras*6].reshape((n_cameras, 6))
            points_3d = params[n_cameras*6:].reshape((n_points, 3))
            projected = []
            points2d = []
            for c in range(0, n_cameras-1):
                for p in range(0, n_points-1):
                    if camera_indices[c] not in trajectories[point_indices[p]]:
                        continue

                    points2d.append(trajectories[point_indices[p]][camera_indices[c]])
                    p3d = points_3d[p]
                    R = camera_params[c][:3]
                    T = camera_params[c][3:]
                    
                    reprojected,_ = cv2.projectPoints(np.array([p3d]), R, T, K, (0,0,0,0))
                    projected.append((reprojected[0][0][0], reprojected[0][0][1]))

            points2d = np.array(points2d)
            projected = np.array(projected)

            return (projected - points2d).ravel()


        x0 = np.hstack((camera_params.ravel(), points_3d_params.ravel()))
        #re = fun(x0, n_cameras, n_points, point_indices, camera_indices, filtered_trajectories, K)
        res = least_squares(fun, x0, args=(n_cameras, n_points, point_indices, camera_indices, filtered_trajectories, K))

        plt.plot(res.fun)
        plt.show()






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

            # Initially, every keypoint is the start of a trajectory.
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
                        # We already have that keypoint, add point for current frame.
                        trajectories[keypoint_no][image_number] = new_keypoints[current_index].pt
                    else:
                        keypoint_no = keypoint_counter
                        keypoint_counter += 1
                        current_index = match.queryIdx
                        next_matches[current_index] = keypoint_no
                        # New keypoint, add points from current and LAST frame, was there already but didn't recognize
                        # it.
                        trajectories[keypoint_no] = {image_number: new_keypoints[current_index].pt,
                                                     image_number - 1: old_keypoints[match.trainIdx].pt}

                current_matches = next_matches
                old_keypoints, old_descriptors = new_keypoints, new_descriptors

                image_number += 1
            else:
                break

        vcap.release()

        return trajectories
