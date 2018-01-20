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

    
    chess_board_rows = 5
    chess_board_columns = 7

    camera_calibration_file = os.path.join("..", "resources", "K.pickle")
    if not os.path.isfile(camera_calibration_file):
        print("Calibrating camera ...")
        max_frame_index = Camera.generate_video_frames(video_file)
        camera_calibration_matrix = Camera.calibrate(max_frame_index, chess_board_rows, chess_board_columns)
        print("The camera has been calibrated.")
        with open(camera_calibration_file, "wb") as handle:
            pickle.dump(camera_calibration_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("The camera is already calibrated.")
        with open(camera_calibration_file, "rb") as handle:
            camera_calibration_matrix = pickle.load(handle)
        print(camera_calibration_matrix)
        camera_calibration_matrix = np.array([[6000.0, 0.0, 536.5],
                                                [0.0, 6000.0, 536.5],
                                                [0.0, 0.0, 1.0]])



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



    cameras, points_3d = Camera.find_keyframe_cameras(trajectories, camera_calibration_matrix, start_frame=1, min_kps=50)
    print(cameras.keys())
    ## Can be commented out for testing trajectories, only little impact on result
    cameras, points_3d = Camera.bundle_adjustment(cameras, trajectories, points_3d, camera_calibration_matrix)

    cameras = Camera.find_all_cameras(trajectories, cameras, points_3d, camera_calibration_matrix)
    

    Renderer.render(cameras, video_file, object3d, object3d_position, object3d_rotation)



if __name__ == '__main__':
    main()

    # original call:
    # main.py -v ..\resources\video.mp4 -o wireframe_cube

    # simplified call:
    # main.py -v ..\resources\video.mp4 -o wireframe_cube


