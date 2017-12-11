# -*- coding: utf-8 -*-

import os
import cv2
from sources.object3d import Object3D


class Renderer(object):
    @staticmethod
    def render(camera_calibration_matrix, video_file, object3d, object3d_position, object3d_rotation, recording):
        recorded_video = os.path.join("..", "recorded", "recorded.avi")
        window_title = "Match-Mover"

        vcap = cv2.VideoCapture(video_file)

        if vcap is None:
            print("The video capture property could not be instantiated.")
            quit()

        # Get the video_frames per second (fps) of the video.
        fps = vcap.get(cv2.CAP_PROP_FPS)

        # Get width and height via video capture property.
        width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if recording:
            # Define the codec and create VideoWriter object.
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            vout = cv2.VideoWriter(recorded_video, fourcc, fps, (int(width), int(height)))

        while vcap.isOpened():
            ret, frame = vcap.read()

            if ret:
                # TODO modify the color here such the video is displayed in its real colors.
                color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # TODO perform match moving here

                cv2.imshow(window_title, color)

                if recording:
                    vout.write(color)
            else:
                break

            key_code = cv2.waitKey(1)

            # Closes the window if the ESC key was pressed.
            if key_code == 27:
                break

            # Closes the window if the X button of the window was clicked.
            if cv2.getWindowProperty(window_title, 1) == -1:
                break

        vcap.release()

        if recording:
            vout.release()

        cv2.destroyAllWindows()
