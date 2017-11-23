#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This Python script plays a given video file with RGB colors. The window can be closed by clicking the "X" button of the
window or by pressing the "ESC" key.
"""

__author__ = "Martin Duregger"
__copyright__ = "Copyright 2017"
__version__ = "1.0.0"

import cv2

WINDOW_NAME = "Video Player"

vcap = cv2.VideoCapture("video.mp4")

if vcap is None:
    quit()

while vcap.isOpened():
    ret, frame = vcap.read()

    if ret:
        color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow(WINDOW_NAME, color)
    else:
        break

    key_code = cv2.waitKey(1)

    # Closes the window if the ESC key was pressed
    if key_code == 27:
        break

    # Closes the window if the X button of the window was clicked
    if cv2.getWindowProperty(WINDOW_NAME, 1) == -1:
        break

vcap.release()
cv2.destroyAllWindows()
