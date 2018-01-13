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
    def render_rectangle(img, camera, points3d):

        nvec = np.cross((points3d[1]-points3d[0]), (points3d[3]-points3d[0]))

        top = []
        for p in points3d:
            top.append(p - nvec/300)

        
        
        points2d,_ = cv2.projectPoints(points3d, camera.R, camera.T, camera.K, (0,0,0,0))

        points2d = [(np.int32(p[0][0]), np.int32(p[0][1])) for p in points2d]
        color = (0,255,0) #bottom, green
        cv2.line(img, points2d[0], points2d[1], color, 10)
        cv2.line(img, points2d[1], points2d[2], color, 10)
        cv2.line(img, points2d[2], points2d[3], color, 10)
        cv2.line(img, points2d[3], points2d[0], color, 10)

        points2d_top,_ = cv2.projectPoints(np.array(top), camera.R, camera.T, camera.K, (0,0,0,0))

        points2d_top = [(np.int32(p[0][0]), np.int32(p[0][1])) for p in points2d_top]
        color = (255,0,0) #bottom, green
        cv2.line(img, points2d_top[0], points2d_top[1], color, 10)
        cv2.line(img, points2d_top[1], points2d_top[2], color, 10)
        cv2.line(img, points2d_top[2], points2d_top[3], color, 10)
        cv2.line(img, points2d_top[3], points2d_top[0], color, 10)

        color = (0,0,255) #bottom, green
        cv2.line(img, points2d[0], points2d_top[0], color, 10)
        cv2.line(img, points2d[1], points2d_top[1], color, 10)
        cv2.line(img, points2d[2], points2d_top[2], color, 10)
        cv2.line(img, points2d[3], points2d_top[3], color, 10)

        return img


    @staticmethod
    def render(img, camera, R):
    #def render(img, R, T, K):
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
        #points3d = Object3D.scale_3d_points(points3d, (10.0, 20.0, 100.0))
        #R,_ = cv2.Rodrigues(rvec)
        #print(R)
        points3d = [np.dot(R.T, p) for p in points3d]

        #move it around
        #points3d = Object3D.scale_3d_points(points3d, (1.0, 1.0, 1.0))
        #points3d = Object3D.translate_3d_points(points3d, -53.0, -8.0, 300.0)

        points3d = cv2.convertPointsFromHomogeneous(np.array(points3d))

        #project using 0 distortion coefficients
        points2d, _ = cv2.projectPoints(points3d, camera.R, camera.T, camera.K, (0,0,0,0))
        #points2d, _ = cv2.projectPoints(np.array(points3d), R, T, K, (0,0,0,0))
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
        
