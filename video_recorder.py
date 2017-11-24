#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This Python script records the video which is currently played. A red circle is also drawn on each frame to test if that
circle is also recorded. The window can be closed by clicking the "X" button of the window or by pressing the "ESC" key.
"""

__author__ = "Martin Duregger"
__copyright__ = "Copyright 2017"
__version__ = "1.0.0"

import cv2

result = "output.avi"

WINDOW_NAME = "Video Recorder"

vcap = cv2.VideoCapture("video.mp4")

# Get the frames per second (fps) of the video
fps = vcap.get(cv2.CAP_PROP_FPS)

# Get width and height via video capture property
width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
vout = cv2.VideoWriter(result, fourcc, fps, (int(width), int(height)))

if vcap is None:
    quit()


numFrames = 0
index = 0
while vcap.isOpened():
    ret, frame = vcap.read()
    if ret:
        color = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if numFrames%20 == 0:

            cv2.imwrite('videoframes/'+str(index)+'.jpg', color)
            index += 1

        cv2.circle(color, (150, 150), 60, (0, 0, 255), 12)

        cv2.imshow(WINDOW_NAME, color)
        vout.write(color)
        numFrames += 1
    else:
        break

    key_code = cv2.waitKey(1)

    # closes the window if the ESC key was pressed
    if key_code == 27:
        break

    # closes the window if the X button of the window was clicked
    if cv2.getWindowProperty(WINDOW_NAME, 1) == -1:
        break

print(numFrames)
vcap.release()
vout.release()
cv2.destroyAllWindows()
