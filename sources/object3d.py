# -*- coding: utf-8 -*-


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
            new_points.append((p[0] * scale, p[1] * scale, p[2] * scale))
        return new_points

    @staticmethod
    def render():
        pass
