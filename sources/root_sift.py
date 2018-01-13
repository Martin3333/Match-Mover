# -*- coding: utf-8 -*-

import cv2
import numpy as np


# https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf
class RootSIFT(object):
    def __init__(self):
        # Initialize the SIFT feature extractor.
        self.extractor = cv2.xfeatures2d.SIFT_create()

    def compute(self, image, kps, eps=1e-7):
        # Compute SIFT descriptors.
        (kps, descs) = self.extractor.compute(image, kps)

        if len(kps) == 0:
            return None, None

        # Apply the Hellinger kernel by first L1-normalizing and taking the square-root.
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)

        return kps, descs
