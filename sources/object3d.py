# -*- coding: utf-8 -*-

import cv2
import numpy as np

class Object3D(object):
    @staticmethod
    def translate_3d_points(points, x, y, z):
        new_points = []
        for p in points:
            new_points.append((p[0] + x, p[1] + y, p[2] + z))
        return new_points

    @staticmethod
    def scale_3d_points(points, scale):
        new_points = []
        for p in points:
            new_points.append((p[0] * scale[0], p[1] * scale[1], p[2] * scale[2]))
        return new_points

    @staticmethod
    def render(img, R, T, K):

        #Simple Cube in 3D
        points3d = []
        points3d.append((1.0, 1.0, 1.0))
        points3d.append((1.0, 1.0, -1.0))
        points3d.append((-1.0, 1.0, -1.0))
        points3d.append((-1.0, 1.0, 1.0))
        points3d.append((1.0, -1.0, 1.0))
        points3d.append((1.0, -1.0, -1.0))
        points3d.append((-1.0, -1.0, -1.0))
        points3d.append((-1.0 , -1.0, 1.0))
        #move it around
        points3d = Object3D.scale_3d_points(points3d, (10.0, 10.0, 100.0))
        points3d = Object3D.translate_3d_points(points3d, 0.0, 0.0, 0.0)

        #project using 0 distortion coefficients
        points2d, _ = cv2.projectPoints(np.array(points3d), R, T, K, (0,0,0,0))
        points2d = [(np.int32(p[0][0]), np.int32(p[0][1])) for p in points2d]

        #draw cube
        color = (255,0,0)   # top, blue
        cv2.line(img, points2d[0], points2d[1], color, 10)
        cv2.line(img, points2d[1], points2d[2], color, 10)
        cv2.line(img, points2d[2], points2d[3], color, 10)
        cv2.line(img, points2d[3], points2d[0], color, 10)
        color = (0,255,0) #bottom, green
        cv2.line(img, points2d[4], points2d[5], color, 10)
        cv2.line(img, points2d[5], points2d[6], color, 10)
        cv2.line(img, points2d[6], points2d[7], color, 10)
        cv2.line(img, points2d[7], points2d[4], color, 10)
        color = (0,0,255) # vertical bars, red
        cv2.line(img, points2d[0], points2d[4], color, 10)
        cv2.line(img, points2d[1], points2d[5], color, 10)
        cv2.line(img, points2d[2], points2d[6], color, 10)
        cv2.line(img, points2d[3], points2d[7], color, 10)


        return img
        
