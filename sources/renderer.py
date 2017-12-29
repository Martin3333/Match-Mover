# -*- coding: utf-8 -*-

import os
import cv2
from object3d import Object3D


class Renderer(object):
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

        frame_index = 0
        while vcap.isOpened():
            ret, frame = vcap.read()

            if ret:
                # TODO modify the color here such the video is displayed in its real colors.
                color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # TODO perform match moving here
                if frame_index in cameras:
                    color = Object3D.render(color, cameras[frame_index])
                    vout.write(color)

                frame_index += 1


                
            else:
                break



        vcap.release()

        if recording:
            vout.release()


