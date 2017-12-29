#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
#import dill
import numpy as np
import getopt
import pickle
import cv2
from camera import Camera
from object3d import Object3D
from renderer import Renderer


def print_usage():
    # TODO write usage text here
    print("Bla bla bla")
    pass


# short program options:
# -v (video), -o (3d object) position[x, y, z] rotation[x, y, z] scale, ..., -r (record playing video), -h (help)
def main():
    # if 6 > len(sys.argv) > 7:  # TODO change number of program arguments here
    #    print_usage()
    #    sys.exit(2)
    try:
        options, arguments = getopt.getopt(sys.argv[1:], "v:o:rh",
                                           ["video=", "object=", "record", "help"])  # TODO maybe some changes here
    except getopt.GetoptError as err:
        print(str(err))
        print_usage()
        sys.exit(2)

    found_v = False
    found_o = False
    recording = False
    video_file = None
    object3d = None
    object3d_position = None
    object3d_rotation = None

    for o, a in options:  # TODO change here the program arguments (also the optional ones)
        if o in ("-v", "--video"):
            if not os.path.isfile(a):
                print("The video file", a, "does not exist.")
                sys.exit(2)

            video_file = a
            found_v = True
        elif o in ("-o", "--object"):
            # TODO perform some checks here
            object3d = a
            found_o = True
        elif o in ("-r", "--record"):
            recording = True
        elif o in ("-h", "--help"):
            print_usage()
            sys.exit()
        else:
            assert False, "unhandled option"

    # Checking if the mandatory options are given.
    if not found_v and not found_o:
        print("options -v and -o were not given")
        print_usage()
        sys.exit(2)
    elif not found_v:
        print("option -v was not given")
        print_usage()
        sys.exit(2)
    elif not found_o:
        print("option -o was not given")
        print_usage()
        sys.exit(2)

    print("Match-Mover started")
    print("============================================")
    print("Video file:", video_file)
    print("3D object:", object3d)
    # print("3D object position (x, y, z):", object3d_position)
    # print("3D object rotation (x-axis, y-axis, z-axis):", object3d_rotation)

    if recording:
        print("Video recording is ON. The recorded video 'recorded.avi' is located in the 'recorded' folder.")
    else:
        print("Video recording is OFF")

    max_frame_index = 91
    chess_board_rows = 5
    chess_board_columns = 7

    camera_calibration_file = os.path.join("..", "resources", "K.pickle")
    if not os.path.isfile(camera_calibration_file):
        print("Calibrating camera ...")
        Camera.generate_video_frames(video_file)
        camera_calibration_matrix = Camera.calibrate(max_frame_index, chess_board_rows, chess_board_columns)
        print("The camera has been calibrated.")
        with open(camera_calibration_file, "wb") as handle:
            pickle.dump(camera_calibration_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("The camera is already calibrated.")
        with open(camera_calibration_file, "rb") as handle:
            camera_calibration_matrix = pickle.load(handle)



    trajectories_file = os.path.join("..", "resources", "trajectories.pickle")
    if not os.path.isfile(trajectories_file):
        print("Detecting point trajectories of video ...")

        trajectories = Camera.detect_trajectories(video_file)
        print("Trajectories detected.")
        with open(trajectories_file, "wb") as handle:
            pickle.dump(trajectories, handle)  
    else:
        print("Trajectories are already detected.")
        with open(trajectories_file, "rb") as handle:
            trajectories = pickle.load(handle)    # TODO check



    cameras = Camera.find_cameras(trajectories, camera_calibration_matrix)

    Renderer.render(cameras, video_file, object3d, object3d_position, object3d_rotation)
    #                 recording)

    # img = cv2.imread('../videoframes/0.jpg', 1)
    # height, width, channels = img.shape
    # recorded_video = os.path.join("..", "recorded", "recorded.mp4")
    # # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Be sure to use lower case
    # out = cv2.VideoWriter(recorded_video, fourcc, 20.0, (width, height))

    # for c in cameras:

    #     img = cv2.imread('../videoframes/' + str(c) + '.jpg', 1)
    #     img = Object3D.render(img, cameras[c])
    #     #cv2.imwrite('../out/' + str(c).zfill(4) + '.jpg', img)
    #     out.write(img)
    # out.release()  



    # R_zero = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    # camera_z_offset = 2000.0


    # start_frame = 1
    # offset, pts1, pts2, indices1 = Camera.find_frames_with_overlap(trajectories, start_frame, min_keypoints=200)
    # offset2, _, pts3, indices2 = Camera.find_frames_with_overlap(trajectories, start_frame, min_keypoints=100)

    # pts1 = np.int32(pts1)
    # pts2 = np.int32(pts2)

    # #TODO : refine points using cv2.correctMatches
    # R,T = Camera.get_camera_pose(pts1, pts2, camera_calibration_matrix)
    # T[2] += camera_z_offset
    
    # P1 = cv2.sfm.projectionFromKRt(camera_calibration_matrix, R_zero, (0.0, 0.0, camera_z_offset))
    # P2 = cv2.sfm.projectionFromKRt(camera_calibration_matrix, R, T)

    # pts1 = [trajectories[i][start_frame] for i in indices2]
    # pts2 = [trajectories[i][start_frame+offset] for i in indices2]

    # object_points = []
    # for p1,p2 in list(zip(pts1, pts2)):
    #     ret = cv2.triangulatePoints(P1, P2, np.array([p1[0],p1[1]]), np.array([p2[0],p2[1]]))
    #     #ret = cv2.convertPointsFromHomogeneous(np.array([ret]))
    #     object_points.append(ret)
    # object_points = cv2.convertPointsFromHomogeneous(np.array(object_points))

    # print(len(object_points))
    # print(len(pts3))
    # ret, R1, T1, _ = cv2.solvePnPRansac(object_points, np.array(pts3), camera_calibration_matrix, (0,0,0,0))


    # img1 = cv2.imread('../videoframes/' + str(start_frame)+'.jpg' ,1)
    # img2 = cv2.imread('../videoframes/' + str(start_frame+offset)+'.jpg' ,1)
    # img3 = cv2.imread('../videoframes/' + str(start_frame+offset2)+'.jpg' ,1)

    # img1 = Object3D.render(img1, (0,0,0), (0,0,camera_z_offset), camera_calibration_matrix)
    # img2 = Object3D.render(img2, R, T, camera_calibration_matrix)
    # img3 = Object3D.render(img3, R1, T1, camera_calibration_matrix)

    # pts1 = [cv2.KeyPoint(p[0], p[1], 50) for p in pts1]
    # pts2 = [cv2.KeyPoint(p[0], p[1], 50) for p in pts2]



    # img1 = cv2.drawKeypoints(img1, pts1, None)
    # img2 = cv2.drawKeypoints(img2, pts2, None)

    # cv2.imshow(str(start_frame), img1)
    # cv2.imshow(str(start_frame+offset), img2)
    # cv2.imshow(str(start_frame+offset2), img3)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



    # print("Start rendering ...")

    # Renderer.render(camera_calibration_matrix, keypoints, descriptors, video_file, object3d, object3d_position, object3d_rotation,
    #                 recording)


if __name__ == '__main__':
    main()

    # original call:
    # main.py -v ..\resources\video.mp4 -o wireframe_cube

    # simplified call:
    # main.py -v ..\resources\video.mp4 -o wireframe_cube


# TODO fix EOFError: Ran out of input in line 114: keypoints = pickle.load(handle)
# TODO fix TypeError: can't pickle cv2.KeyPoint objects
