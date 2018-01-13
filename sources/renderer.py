# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from object3d import Object3D


class Renderer(object):
    @staticmethod
    def find_chessboard_location(cameras):

        #pts1 = [(400.0, 502.0), (494.0, 435.0), (593.0, 494.0), (501.0, 567.0)]
        pts1 = [(403.0, 503.0), (498.0, 435.0), (598.0, 493.0), (505.0, 569.0)] # frame 10
        pts2 = [(449.0, 471.0), (584.0, 457.0), (606.0, 554.0), (459.0, 571.0)] # frame 180
        #pts2 = [(609.0, 348.0),(739.0, 375.0), (699.0, 468.0), (561.0, 435.0)] # frame 300
        #print(cameras[20].R)
        #print(cameras.keys())
        P1 = cameras[10].P_from_RT()
        P2 = cameras[180].P_from_RT()

        object_points = []
        for p1,p2 in list(zip(pts1, pts2)):
            ret = cv2.triangulatePoints(P1, P2, np.array([p1[0],p1[1]]), np.array([p2[0],p2[1]]))
            object_points.append(ret)
        object_points = cv2.convertPointsFromHomogeneous(np.array(object_points))


        #return trans_matrix[1]
        return object_points


    @staticmethod
    def render(cameras, video_file, object3d, object3d_position, object3d_rotation,
               recording = True):
        recorded_video = os.path.join("..", "recorded", "recorded.avi")
        window_title = "Match-Mover"

        vcap = cv2.VideoCapture(video_file)

        if vcap is None or not vcap.isOpened():
            print("The video capture property could not be instantiated.")
            quit()

        # Get the video_frames per second (fps) of the video.
        fps = vcap.get(cv2.CAP_PROP_FPS)

        # Get width and height via video capture property.
        width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)


        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vout = cv2.VideoWriter(recorded_video, fourcc, fps, (int(width), int(height)))
        first = True
        frame_index = 0
        while vcap.isOpened():
            ret, frame = vcap.read()

            if ret:



                # TODO modify the color here such the video is displayed in its real colors.
                color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # TODO perform match moving here
                if frame_index in cameras:
                    if first:
                        rvec = Renderer.find_chessboard_location(cameras)
                        first = False

                    color = Object3D.render_rectangle(color, cameras[frame_index], rvec)
                    vout.write(color)

                frame_index += 1


                
            else:
                break



        vcap.release()

        if recording:
            vout.release()



            

